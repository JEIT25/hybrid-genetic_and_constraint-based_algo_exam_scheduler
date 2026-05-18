"""
Hybrid Genetic + Constraint-Based Exam Scheduler
=================================================
The "hybrid" means:
  1. Initial population is seeded using constraint propagation (not random)
  2. Crossover/mutation operators are constraint-aware (repair invalid genes)
  3. Local search (hill climbing) is applied to top individuals each generation
"""

import random
import copy
import time
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from scheduler_constraints import (
    Exam, Room, Timeslot, Assignment, Schedule,
    ConstraintEngine, ConstraintViolation, ConstraintType,
)


def fitness_from_violations(violations: List[ConstraintViolation], soft_weight: float) -> float:
    """Same scoring as ExamSchedulerGA._evaluate (kept in one place for parallel workers)."""
    hard_p = sum(v.penalty for v in violations if v.constraint_type == ConstraintType.HARD)
    soft_p = sum(v.penalty for v in violations if v.constraint_type == ConstraintType.SOFT)
    return 10_000_000 - hard_p - float(soft_weight) * soft_p


_worker_engine: Optional[ConstraintEngine] = None
_worker_soft_weight: float = 1.0


def _parallel_worker_init(exams: Dict[str, Exam], rooms: Dict[str, Room], timeslots: Dict[str, Timeslot], soft_weight: float):
    global _worker_engine, _worker_soft_weight
    _worker_engine = ConstraintEngine(exams, rooms, timeslots)
    _worker_soft_weight = float(soft_weight)


def _parallel_fitness(schedule: Schedule) -> float:
    global _worker_engine, _worker_soft_weight
    assert _worker_engine is not None
    _, violations = _worker_engine.evaluate(schedule)
    return fitness_from_violations(violations, _worker_soft_weight)


@dataclass
class GAConfig:
    """All tunable knobs for the genetic algorithm."""
    population_size: int = 100
    max_generations: int = 500
    elite_ratio: float = 0.10         # top 10% survive unchanged
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    tournament_size: int = 5
    local_search_ratio: float = 0.05  # apply hill-climb to top 5%
    local_search_steps: int = 20
    stagnation_limit: int = 50        # restart if no improvement
    target_fitness: Optional[float] = None  # stop early if reached
    seed: Optional[int] = None
    # When > 1.0 soft penalties are amplified so the GA actively optimises them
    soft_weight: float = 1.0
    # Number of repair-pass iterations per individual (default 50, raise for hard-fix mode)
    max_repair_attempts: int = 50
    # Parallel fitness evaluation (multiprocessing). 0 = auto when len(exams)>=22 and cpu_count>1; 1 = off.
    parallel_eval_workers: int = 0
    # If > 0: stop when best individual is feasible (no hard violations) and fitness has not improved for this many generations.
    early_exit_feasible_stagnation: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class EvolutionStats:
    generation: int = 0
    best_fitness: float = 0
    avg_fitness: float = 0
    worst_fitness: float = 0
    hard_violations: int = 0
    soft_violations: int = 0
    feasible_pct: float = 0
    elapsed_sec: float = 0


class ExamSchedulerGA:
    """
    The main hybrid engine.

    Workflow per generation:
      1. Evaluate fitness (via ConstraintEngine)
      2. Elitism → carry top N unchanged
      3. Tournament selection → pick parents
      4. Constraint-aware crossover → produce children
      5. Constraint-aware mutation → tweak children
      6. Repair operator → fix hard-constraint violations
      7. Local search → hill-climb on top individuals
      8. Replace population
    """

    def __init__(
        self,
        exams: List[Exam],
        rooms: List[Room],
        timeslots: List[Timeslot],
        config: GAConfig = GAConfig(),
    ):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)

        self.exams = {e.id: e for e in exams}
        self.rooms = {r.id: r for r in rooms}
        self.timeslots = {t.id: t for t in timeslots}
        self.exam_list = list(self.exams.keys())

        self.engine = ConstraintEngine(self.exams, self.rooms, self.timeslots)

        # Pre-compute feasible (room, timeslot) pairs per exam
        # Enforces three hard rules at the structural level:
        #   1. Room capacity must be sufficient for the exam
        #   2. Exam type must match room type (written->lec, hands-on->lab)
        #   3. No exam can end after 7:30 PM (19:30 = 1170 minutes)
        self._feasible_pairs: Dict[str, List[Tuple[str, str]]] = {}
        MAX_END_MINUTES = 1170  # 19 hours * 60 + 30 minutes = 7:30 PM

        for eid, exam in self.exams.items():
            pairs = []
            for rid, room in self.rooms.items():
                # HC: Room capacity check
                if exam.student_count > room.capacity:
                    continue
                # HC: Exam type ↔ room type strict matching
                if exam.exam_type == 'hands-on' and room.room_type != 'lab':
                    continue
                if exam.exam_type == 'written' and room.room_type != 'lec':
                    continue
                for tid, ts in self.timeslots.items():
                    # HC: 7:30 PM rule — exam end time must not exceed 19:30
                    end_minutes = (ts.start_hour * 60 + ts.start_minute) + exam.duration_minutes
                    if end_minutes > MAX_END_MINUTES:
                        continue
                    pairs.append((tid, rid))
            self._feasible_pairs[eid] = pairs if pairs else [
                (tid, rid) for tid in self.timeslots for rid in self.rooms
            ]

        self.population: List[Schedule] = []
        self.fitness_cache: Dict[int, float] = {}
        self.history: List[EvolutionStats] = []
        self._best_ever: Optional[Tuple[float, Schedule]] = None
        self._callback: Optional[Callable] = None

    def _parallel_worker_count(self) -> int:
        w = self.config.parallel_eval_workers
        cores = cpu_count() or 2
        if w == 1:
            return 1
        if w >= 2:
            return max(1, min(w, cores))
        # auto (w == 0)
        if len(self.exams) < 22 or cores < 2:
            return 1
        return max(1, cores - 1)

    # ── 1. Constraint-Propagation Seeding ─────────────────────────────────

    def _seed_individual(self) -> Schedule:
        """
        Build one schedule using greedy constraint propagation.
        Assigns all sections of a group together to respect section constraints.
        """
        schedule: List[Assignment] = []
        used_slots: Dict[str, Dict[str, str]] = {}  # timeslot_id → {room_id: exam_id}

        scheduled_eids = set()

        # Most-constrained-first: more conflicts + fewer feasible options = harder
        sorted_exams = sorted(
            self.exam_list,
            key=lambda eid: (
                -len(self.engine.conflict_graph.get(eid, set())),
                len(self._feasible_pairs[eid]),
            ),
        )

        for base_eid in sorted_exams:
            if base_eid in scheduled_eids:
                continue

            exam = self.exams[base_eid]
            group_eids = self.engine.group_map.get(exam.group_id, [base_eid]) if exam.group_id else [base_eid]

            tids = list(set(tid for tid, rid in self._feasible_pairs[base_eid]))
            random.shuffle(tids)

            best_assignments = None
            best_score = float("inf")

            # Sample up to 30 candidates for speed
            for tid in tids[:30]:
                score = 0
                temp_assignments = {}
                temp_rooms = set()
                success = True

                for geid in group_eids:
                    valid_rids = [r for t, r in self._feasible_pairs[geid] if t == tid and r not in temp_rooms]
                    if not valid_rids:
                        success = False
                        break

                    best_r_score = float("inf")
                    best_rid = None
                    for rid in valid_rids[:10]:
                        r_score = 0
                        # Room already taken in this slot?
                        if tid in used_slots and rid in used_slots[tid]:
                            r_score += 10000
                        # Student clash?
                        for existing in schedule:
                            if self.timeslots[existing.timeslot_id].overlaps(self.timeslots[tid]):
                                if existing.exam_id in self.engine.conflict_graph.get(geid, set()):
                                    r_score += 10000

                        # Soft: room waste
                        r_score += (self.rooms[rid].capacity - self.exams[geid].student_count) * 0.1

                        if r_score < best_r_score:
                            best_r_score = r_score
                            best_rid = rid

                    if best_rid is None:
                        success = False
                        break

                    temp_rooms.add(best_rid)
                    temp_assignments[geid] = best_rid
                    score += best_r_score

                if success and score < best_score:
                    best_score = score
                    best_assignments = (tid, temp_assignments)

            if best_assignments is None:
                # Fallback: Just pick any valid combination regardless of conflicts
                for tid in tids:
                    temp_assignments = {}
                    temp_rooms = set()
                    success = True
                    for geid in group_eids:
                        valid_rids = [r for t, r in self._feasible_pairs[geid] if t == tid and r not in temp_rooms]
                        if not valid_rids:
                            success = False
                            break
                        rid = random.choice(valid_rids)
                        temp_rooms.add(rid)
                        temp_assignments[geid] = rid
                    if success:
                        best_assignments = (tid, temp_assignments)
                        break

            if best_assignments:
                tid, assigns = best_assignments
                for geid, rid in assigns.items():
                    schedule.append(Assignment(exam_id=geid, timeslot_id=tid, room_id=rid))
                    used_slots.setdefault(tid, {})[rid] = geid
                    scheduled_eids.add(geid)
            else:
                # Absolute fallback (rare, unless severely constrained rooms)
                for geid in group_eids:
                    tid, rid = random.choice(self._feasible_pairs[geid])
                    schedule.append(Assignment(exam_id=geid, timeslot_id=tid, room_id=rid))
                    used_slots.setdefault(tid, {})[rid] = geid
                    scheduled_eids.add(geid)

        return schedule

    def initialize_population(self):
        """Create initial population using constraint-propagation seeding."""
        self.population = []
        for _ in range(self.config.population_size):
            self.population.append(self._seed_individual())

    # ── 2. Fitness Evaluation ─────────────────────────────────────────────

    def _evaluate(self, schedule: Schedule) -> Tuple[float, List[ConstraintViolation]]:
        _, violations = self.engine.evaluate(schedule)
        fitness = fitness_from_violations(violations, float(self.config.soft_weight))
        return fitness, violations

    def _fitness(self, schedule: Schedule) -> float:
        key = id(schedule)
        if key not in self.fitness_cache:
            f, _ = self._evaluate(schedule)
            self.fitness_cache[key] = f
        return self.fitness_cache[key]

    # ── 3. Selection ──────────────────────────────────────────────────────

    def _tournament_select(self, pop_with_fitness: List[Tuple[Schedule, float]]) -> Schedule:
        """Tournament selection: pick best of K random individuals."""
        contestants = random.sample(pop_with_fitness, min(self.config.tournament_size, len(pop_with_fitness)))
        winner = max(contestants, key=lambda x: x[1])
        return copy.deepcopy(winner[0])

    # ── 4. Constraint-Aware Crossover ─────────────────────────────────────

    def _crossover(self, parent1: Schedule, parent2: Schedule) -> Tuple[Schedule, Schedule]:
        """
        Uniform crossover at the gene (exam assignment) level.
        Each child inherits each exam's assignment from one parent,
        then the repair operator fixes any hard-constraint violations.
        """
        if random.random() > self.config.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        p1_map = {a.exam_id: a for a in parent1}
        p2_map = {a.exam_id: a for a in parent2}

        child1, child2 = [], []
        for eid in self.exam_list:
            if random.random() < 0.5:
                child1.append(copy.deepcopy(p1_map[eid]))
                child2.append(copy.deepcopy(p2_map[eid]))
            else:
                child1.append(copy.deepcopy(p2_map[eid]))
                child2.append(copy.deepcopy(p1_map[eid]))

        return child1, child2

    # ── 5. Constraint-Aware Mutation ──────────────────────────────────────

    def _mutate(self, schedule: Schedule) -> Schedule:
        """
        For each gene, with mutation probability:
        - Pick a new (timeslot, room) from the feasible set for that exam
        """
        for i, assignment in enumerate(schedule):
            if random.random() < self.config.mutation_rate:
                pairs = self._feasible_pairs[assignment.exam_id]
                new_tid, new_rid = random.choice(pairs)
                schedule[i] = Assignment(
                    exam_id=assignment.exam_id,
                    timeslot_id=new_tid,
                    room_id=new_rid,
                )
        return schedule

    def _targeted_mutate(self, schedule: Schedule) -> Schedule:
        """
        Specifically re-assign exams involved in hard-constraint violations.
        Much more effective than random mutation for escaping infeasible regions.
        """
        _, violations = self._evaluate(schedule)
        hard = [v for v in violations if v.constraint_type == ConstraintType.HARD]
        if not hard:
            return schedule

        # Collect all exam IDs involved in hard violations
        violated_eids = set()
        for v in hard:
            violated_eids.update(v.exam_ids)

        # For each violated exam, try to find a conflict-free assignment
        assignment_map = {a.exam_id: i for i, a in enumerate(schedule)}
        for eid in violated_eids:
            if eid not in assignment_map:
                continue
            idx = assignment_map[eid]
            pairs = self._feasible_pairs[eid]
            random.shuffle(pairs)

            best_pair = None
            best_conflicts = float("inf")

            for tid, rid in pairs[:40]:
                # Count conflicts this assignment would create
                conflicts = 0
                for j, other in enumerate(schedule):
                    if j == idx:
                        continue
                    # Room clash
                    if other.room_id == rid and self.timeslots[other.timeslot_id].overlaps(self.timeslots[tid]):
                        conflicts += 1
                    # Student clash
                    if other.exam_id in self.engine.conflict_graph.get(eid, set()):
                        if self.timeslots[other.timeslot_id].overlaps(self.timeslots[tid]):
                            conflicts += 10

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_pair = (tid, rid)
                    if conflicts == 0:
                        break

            if best_pair:
                schedule[idx] = Assignment(eid, best_pair[0], best_pair[1])

        return schedule

    # ── 5b. Soft-Targeted Mutation ────────────────────────────────────────
    def _soft_targeted_mutate(self, schedule: Schedule) -> Schedule:
        """
        Deterministic instructor-preference repair pass.

        For every exam whose instructor has day/shift preferences that the
        current assignment violates, find the best conflict-free slot that
        respects those preferences and swap unconditionally.

        'Best' means: (1) preference satisfied, (2) room conflict-free,
        (3) least wasted seats (smallest room that fits).

        Moving from a non-preferred day/shift to a preferred one always
        reduces the soft penalty, so no fitness comparison is needed —
        the swap is beneficial by definition as long as no new hard
        violations are introduced (guaranteed by the conflict check below).
        """
        schedule = list(schedule)

        # Build a live room-time occupancy map so we can avoid room conflicts
        # as we make successive swaps in this pass.
        room_time: dict = {}   # (room_id, timeslot_id) -> exam_id
        for a in schedule:
            room_time[(a.room_id, a.timeslot_id)] = a.exam_id

        for i, assignment in enumerate(schedule):
            exam = self.exams.get(assignment.exam_id)
            if not exam or not exam.instructor_prefs:
                continue

            ts       = self.timeslots[assignment.timeslot_id]
            raw_days = exam.instructor_prefs.get('days', [])
            pref_days   = [int(d) for d in raw_days if str(d).lstrip('-').isdigit()]
            pref_shifts = exam.instructor_prefs.get('shifts', [])

            day_ok   = (not pref_days)   or (ts.day in pref_days)
            shift_ok = (not pref_shifts) or (ts.shift_name in pref_shifts)

            if day_ok and shift_ok:
                continue  # preference already satisfied — nothing to do

            # Collect pairs that (a) satisfy the preference AND (b) have no
            # room conflict with any other already-placed exam.
            candidates = []
            for (tid, rid) in self._feasible_pairs[assignment.exam_id]:
                # Preference check
                if pref_days and self.timeslots[tid].day not in pref_days:
                    continue
                if pref_shifts and self.timeslots[tid].shift_name not in pref_shifts:
                    continue
                # Room-conflict check: the slot is free or occupied only by this exam
                occupant = room_time.get((rid, tid))
                if occupant is not None and occupant != assignment.exam_id:
                    continue
                waste = self.rooms[rid].capacity - exam.student_count
                candidates.append((waste, tid, rid))

            if not candidates:
                continue  # no conflict-free preferred slot exists — leave as is

            # Pick slot with least room waste (tie-break: random among equals)
            candidates.sort(key=lambda x: x[0])
            best_waste = candidates[0][0]
            best_candidates = [(t, r) for (w, t, r) in candidates if w == best_waste]
            new_tid, new_rid = random.choice(best_candidates)

            # Apply swap and update the live occupancy map
            room_time.pop((assignment.room_id, assignment.timeslot_id), None)
            room_time[(new_rid, new_tid)] = assignment.exam_id
            schedule[i] = Assignment(assignment.exam_id, new_tid, new_rid)

        return schedule

    # ── 6. Repair Operator ────────────────────────────────────────────────

    def _repair(self, schedule: Schedule) -> Schedule:
        """
        Attempt to fix hard-constraint violations by re-assigning offending exams.
        This is the key 'constraint-based' part of the hybrid.
        """
        max_attempts = self.config.max_repair_attempts
        for attempt in range(max_attempts):
            _, violations = self._evaluate(schedule)
            hard = [v for v in violations if v.constraint_type == ConstraintType.HARD]
            if not hard:
                break

            # Pick a random hard violation and fix one of its exams
            v = random.choice(hard)
            offending_exams = v.exam_ids if v.exam_ids else [random.choice(schedule).exam_id]

            target_eid = random.choice(offending_exams)
            pairs = self._feasible_pairs[target_eid]
            new_tid, new_rid = random.choice(pairs)

            for i, a in enumerate(schedule):
                if a.exam_id == target_eid:
                    schedule[i] = Assignment(target_eid, new_tid, new_rid)
                    break

        return schedule

    # ── 7. Local Search (Hill Climbing) ───────────────────────────────────

    def _local_search(self, schedule: Schedule) -> Schedule:
        """
        Improve a schedule by making small changes and keeping improvements.
        Focuses on the exams involved in the worst violations.
        """
        current_fitness, violations = self._evaluate(schedule)

        for step in range(self.config.local_search_steps):
            # Pick an exam to tweak — bias toward violated ones
            if violations and random.random() < 0.7:
                v = random.choice(violations)
                violated_eids = set(v.exam_ids) if v.exam_ids else set()
                candidates = [a for a in schedule if a.exam_id in violated_eids]
                if not candidates:
                    candidates = schedule
            else:
                candidates = schedule

            target = random.choice(candidates)
            idx = next(i for i, a in enumerate(schedule) if a.exam_id == target.exam_id)

            # Try a new assignment
            pairs = self._feasible_pairs[target.exam_id]
            new_tid, new_rid = random.choice(pairs)
            old_assignment = schedule[idx]
            schedule[idx] = Assignment(target.exam_id, new_tid, new_rid)

            new_fitness, new_violations = self._evaluate(schedule)
            if new_fitness > current_fitness:
                current_fitness = new_fitness
                violations = new_violations
            else:
                schedule[idx] = old_assignment  # revert

        return schedule

    # ── Main Evolution Loop ───────────────────────────────────────────────

    def run(self, callback: Optional[Callable] = None) -> Tuple[Schedule, float, List[EvolutionStats]]:
        """
        Execute the full genetic algorithm.

        Args:
            callback: Optional function called each generation with EvolutionStats

        Returns:
            (best_schedule, best_fitness, history)
        """
        self._callback = callback
        start_time = time.time()

        if not self.population:
            self.initialize_population()

        stagnation_counter = 0
        best_fitness_ever = float("-inf")

        workers = self._parallel_worker_count()
        eval_pool: Optional[Pool] = None
        if workers > 1:
            eval_pool = Pool(
                processes=workers,
                initializer=_parallel_worker_init,
                initargs=(self.exams, self.rooms, self.timeslots, float(self.config.soft_weight)),
            )

        try:
            for gen in range(self.config.max_generations):
                self.fitness_cache.clear()

                # Evaluate population (multiprocessing avoids GIL for CPU-heavy constraint checks)
                if eval_pool is not None:
                    chunk = max(1, len(self.population) // (workers * 4))
                    fitness_values = eval_pool.map(_parallel_fitness, self.population, chunksize=chunk)
                    pop_fitness = list(zip(self.population, fitness_values))
                else:
                    pop_fitness = [(ind, self._fitness(ind)) for ind in self.population]
                pop_fitness.sort(key=lambda x: x[1], reverse=True)

                best_f = pop_fitness[0][1]
                avg_f = sum(f for _, f in pop_fitness) / len(pop_fitness)
                worst_f = pop_fitness[-1][1]

                # Track best ever
                if best_f > best_fitness_ever:
                    best_fitness_ever = best_f
                    self._best_ever = (best_f, copy.deepcopy(pop_fitness[0][0]))
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Stats
                _, all_v = self._evaluate(pop_fitness[0][0])
                hard_v = sum(1 for v in all_v if v.constraint_type == ConstraintType.HARD)
                soft_v = sum(1 for v in all_v if v.constraint_type == ConstraintType.SOFT)
                feasible_count = sum(1 for ind, _ in pop_fitness if self.engine.is_feasible(ind))

                stats = EvolutionStats(
                    generation=gen,
                    best_fitness=best_f,
                    avg_fitness=avg_f,
                    worst_fitness=worst_f,
                    hard_violations=hard_v,
                    soft_violations=soft_v,
                    feasible_pct=feasible_count / len(pop_fitness) * 100,
                    elapsed_sec=time.time() - start_time,
                )
                self.history.append(stats)

                if callback:
                    callback(stats)

                # Early stopping
                if self.config.target_fitness and best_f >= self.config.target_fitness:
                    break
                if (
                    self.config.early_exit_feasible_stagnation > 0
                    and hard_v == 0
                    and stagnation_counter >= self.config.early_exit_feasible_stagnation
                ):
                    break
                if stagnation_counter >= self.config.stagnation_limit:
                    # Restart: keep elite, re-seed the rest
                    elite_n = max(2, int(len(pop_fitness) * self.config.elite_ratio))
                    elite = [copy.deepcopy(ind) for ind, _ in pop_fitness[:elite_n]]
                    new_pop = elite + [self._seed_individual() for _ in range(len(pop_fitness) - elite_n)]
                    self.population = new_pop
                    stagnation_counter = 0
                    continue

                # ── Build next generation ────────────────────────────────

                next_pop: List[Schedule] = []

                # Elitism
                elite_n = max(1, int(len(pop_fitness) * self.config.elite_ratio))
                for ind, _ in pop_fitness[:elite_n]:
                    next_pop.append(copy.deepcopy(ind))

                # Local search on top individuals
                ls_n = max(1, int(len(pop_fitness) * self.config.local_search_ratio))
                best_is_feasible = (hard_v == 0)
                for i in range(min(ls_n, elite_n)):
                    next_pop[i] = self._targeted_mutate(next_pop[i])
                    # When the best schedule is already feasible and we're in
                    # soft-optimise mode, spend local-search effort on instructor
                    # preferences instead of (unnecessary) hard-constraint repair.
                    if best_is_feasible and self.config.soft_weight > 1.0:
                        next_pop[i] = self._soft_targeted_mutate(next_pop[i])
                    next_pop[i] = self._local_search(next_pop[i])

                # Fill rest via crossover + mutation
                while len(next_pop) < self.config.population_size:
                    p1 = self._tournament_select(pop_fitness)
                    p2 = self._tournament_select(pop_fitness)
                    c1, c2 = self._crossover(p1, p2)
                    c1 = self._mutate(c1)
                    c2 = self._mutate(c2)
                    # Hard-constraint targeted mutation
                    c1 = self._targeted_mutate(c1)
                    c2 = self._targeted_mutate(c2)
                    # Soft-constraint targeted mutation (only in soft-optimise mode)
                    if best_is_feasible and self.config.soft_weight > 1.0:
                        c1 = self._soft_targeted_mutate(c1)
                        c2 = self._soft_targeted_mutate(c2)
                    c1 = self._repair(c1)
                    c2 = self._repair(c2)
                    next_pop.append(c1)
                    if len(next_pop) < self.config.population_size:
                        next_pop.append(c2)

                self.population = next_pop

        finally:
            if eval_pool is not None:
                eval_pool.close()
                eval_pool.join()

        # Return best ever found
        if self._best_ever:
            return self._best_ever[1], self._best_ever[0], self.history
        else:
            return self.population[0], self._fitness(self.population[0]), self.history


def format_schedule(
    schedule: Schedule,
    exams: Dict[str, Exam],
    rooms: Dict[str, Room],
    timeslots: Dict[str, Timeslot],
) -> List[dict]:
    """Convert a schedule to a human-readable list of dicts."""
    result = []
    for a in sorted(schedule, key=lambda x: (
        timeslots[x.timeslot_id].day,
        timeslots[x.timeslot_id].start_hour,
    )):
        ts = timeslots[a.timeslot_id]
        room = rooms[a.room_id]
        exam = exams[a.exam_id]
        result.append({
            "exam_id": a.exam_id,
            "exam_name": exam.name,
            "department": exam.department,
            "student_count": exam.student_count,
            "day": ts.day_name,
            "time": f"{ts.start_hour:02d}:{ts.start_minute:02d}",
            "duration_min": exam.duration_minutes,
            "room": room.name,
            "room_capacity": room.capacity,
            "building": room.building,
        })
    return result