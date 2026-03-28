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
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from scheduler_constraints import (
    Exam, Room, Timeslot, Assignment, Schedule,
    ConstraintEngine, ConstraintViolation, ConstraintType,
)


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
        self._feasible_pairs: Dict[str, List[Tuple[str, str]]] = {}
        for eid, exam in self.exams.items():
            pairs = []
            for rid, room in self.rooms.items():
                if exam.student_count > room.capacity:
                    continue
                if exam.requires_computer and not room.has_computers:
                    continue
                for tid in self.timeslots:
                    pairs.append((tid, rid))
            self._feasible_pairs[eid] = pairs if pairs else [
                (tid, rid) for tid in self.timeslots for rid in self.rooms
            ]

        self.population: List[Schedule] = []
        self.fitness_cache: Dict[int, float] = {}
        self.history: List[EvolutionStats] = []
        self._best_ever: Optional[Tuple[float, Schedule]] = None
        self._callback: Optional[Callable] = None

    # ── 1. Constraint-Propagation Seeding ─────────────────────────────────

    def _seed_individual(self) -> Schedule:
        """
        Build one schedule using greedy constraint propagation:
        - Sort exams by difficulty (most constrained first)
        - For each exam, pick a (timeslot, room) that causes fewest conflicts
        """
        schedule: List[Assignment] = []
        used_slots: Dict[str, Dict[str, str]] = {}  # timeslot_id → {room_id: exam_id}

        # Most-constrained-first: more conflicts + fewer feasible options = harder
        sorted_exams = sorted(
            self.exam_list,
            key=lambda eid: (
                -len(self.engine.conflict_graph.get(eid, set())),
                len(self._feasible_pairs[eid]),
            ),
        )

        for eid in sorted_exams:
            candidates = self._feasible_pairs[eid]
            random.shuffle(candidates)

            best_pair = None
            best_score = float("inf")

            # Sample up to 30 candidates for speed
            for tid, rid in candidates[:30]:
                score = 0
                # Room already taken in this slot?
                if tid in used_slots and rid in used_slots[tid]:
                    score += 10000
                # Student clash?
                for existing in schedule:
                    if self.timeslots[existing.timeslot_id].overlaps(self.timeslots[tid]):
                        if existing.exam_id in self.engine.conflict_graph.get(eid, set()):
                            score += 10000

                # Soft: room waste
                room = self.rooms[rid]
                exam = self.exams[eid]
                score += (room.capacity - exam.student_count) * 0.1

                if score < best_score:
                    best_score = score
                    best_pair = (tid, rid)

            if best_pair is None:
                best_pair = random.choice(candidates)

            tid, rid = best_pair
            schedule.append(Assignment(exam_id=eid, timeslot_id=tid, room_id=rid))
            used_slots.setdefault(tid, {})[rid] = eid

        return schedule

    def initialize_population(self):
        """Create initial population using constraint-propagation seeding."""
        self.population = []
        for _ in range(self.config.population_size):
            self.population.append(self._seed_individual())

    # ── 2. Fitness Evaluation ─────────────────────────────────────────────

    def _evaluate(self, schedule: Schedule) -> Tuple[float, List[ConstraintViolation]]:
        return self.engine.evaluate(schedule)

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

    # ── 6. Repair Operator ────────────────────────────────────────────────

    def _repair(self, schedule: Schedule) -> Schedule:
        """
        Attempt to fix hard-constraint violations by re-assigning offending exams.
        This is the key 'constraint-based' part of the hybrid.
        """
        max_attempts = 50
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

        for gen in range(self.config.max_generations):
            self.fitness_cache.clear()

            # Evaluate all
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
            for i in range(min(ls_n, elite_n)):
                next_pop[i] = self._targeted_mutate(next_pop[i])
                next_pop[i] = self._local_search(next_pop[i])

            # Fill rest via crossover + mutation
            while len(next_pop) < self.config.population_size:
                p1 = self._tournament_select(pop_fitness)
                p2 = self._tournament_select(pop_fitness)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                # Targeted mutation to directly address hard violations
                c1 = self._targeted_mutate(c1)
                c2 = self._targeted_mutate(c2)
                c1 = self._repair(c1)
                c2 = self._repair(c2)
                next_pop.append(c1)
                if len(next_pop) < self.config.population_size:
                    next_pop.append(c2)

            self.population = next_pop

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