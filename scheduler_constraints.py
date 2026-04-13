"""
Constraint Engine for Exam Scheduling
======================================
Updated for Laravel integration with instructor assignments.

Hard constraints → must NEVER be violated
Soft constraints → penalize the score but still produce a valid schedule
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Exam:
    id: str
    name: str
    duration_minutes: int
    student_count: int
    enrolled_students: Set[str] = field(default_factory=set)
    department: str = ""
    requires_computer: bool = False
    priority: int = 0
    instructor_id: Optional[int] = None  # Laravel user_id
    exam_type: str = "written"            # 'written' or 'hands-on'
    instructor_prefs: dict = field(default_factory=dict)  # {'days': [0,1], 'shifts': ['morning']}
    group_id: str = ""    # Links sections of the same subject (e.g. "subj_5")
    section_id: str = ""  # The specific section this exam represents


@dataclass
class Room:
    id: str
    name: str
    capacity: int
    has_computers: bool = False
    building: str = ""
    room_type: str = "lec"  # 'lec' or 'lab'


@dataclass
class Timeslot:
    id: str
    day: int        # 0 = Monday, 1 = Tuesday, ... 4 = Friday
    start_hour: int
    start_minute: int = 0
    duration_minutes: int = 120
    date_str: str = ""  # Actual date string from Laravel (e.g. "2026-04-01")

    @property
    def end_time_minutes(self) -> int:
        """Calculate the ending time in total minutes from midnight."""
        return (self.start_hour * 60) + self.start_minute + self.duration_minutes

    @property
    def shift_name(self) -> str:
        """
        Determine which shift this timeslot belongs to:
          - Starts before 12:00 → 'morning'
          - Starts 12:00 to 16:59 → 'afternoon'
          - Starts 17:00 or later → 'evening'
        """
        start = self.start_hour * 60 + self.start_minute
        if start < 720:      # before 12:00
            return "morning"
        elif start < 1020:   # before 17:00
            return "afternoon"
        else:
            return "evening"

    @property
    def day_name(self) -> str:
        """Return the short weekday name for this timeslot's day."""
        return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][self.day]

    def overlaps(self, other: "Timeslot") -> bool:
        if self.day != other.day:
            return False
        s1 = self.start_hour * 60 + self.start_minute
        e1 = s1 + self.duration_minutes
        s2 = other.start_hour * 60 + other.start_minute
        e2 = s2 + other.duration_minutes
        return s1 < e2 and s2 < e1

    def is_consecutive(self, other: "Timeslot") -> bool:
        if self.day != other.day:
            return False
        e1 = (self.start_hour * 60 + self.start_minute) + self.duration_minutes
        s2 = other.start_hour * 60 + other.start_minute
        e2 = s2 + other.duration_minutes
        s1 = self.start_hour * 60 + self.start_minute
        return abs(e1 - s2) <= 30 or abs(e2 - s1) <= 30


@dataclass
class Assignment:
    exam_id: str
    timeslot_id: str
    room_id: str


Schedule = List[Assignment]


# ─── Constraint Definitions ───────────────────────────────────────────────────

class ConstraintType(Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class ConstraintViolation:
    constraint_name: str
    constraint_type: ConstraintType
    penalty: float
    description: str
    exam_ids: List[str] = field(default_factory=list)


class ConstraintEngine:
    HARD_PENALTY = 1_000_000

    def __init__(self, exams, rooms, timeslots):
        self.exams = exams
        self.rooms = rooms
        self.timeslots = timeslots

        # Pre-compute student conflict graph
        self.conflict_graph: Dict[str, Set[str]] = {eid: set() for eid in exams}
        exam_list = list(exams.values())
        for i in range(len(exam_list)):
            for j in range(i + 1, len(exam_list)):
                if exam_list[i].enrolled_students & exam_list[j].enrolled_students:
                    self.conflict_graph[exam_list[i].id].add(exam_list[j].id)
                    self.conflict_graph[exam_list[j].id].add(exam_list[i].id)

        # Pre-compute instructor conflict graph
        self.instructor_map: Dict[str, List[str]] = {}  # instructor_id → [exam_ids]
        for eid, exam in exams.items():
            if exam.instructor_id:
                iid = str(exam.instructor_id)
                self.instructor_map.setdefault(iid, []).append(eid)

        # Pre-compute group map (sections of the same subject)
        # group_id → [exam_ids] — used by same-timeslot and different-room constraints
        self.group_map: Dict[str, List[str]] = {}
        for eid, exam in exams.items():
            if exam.group_id:
                self.group_map.setdefault(exam.group_id, []).append(eid)

    # ── Hard Constraints ──────────────────────────────────────────────────

    def _check_student_clash(self, schedule):
        violations = []
        slot_map = {}
        for a in schedule:
            slot_map.setdefault(a.timeslot_id, []).append(a.exam_id)

        checked = set()
        for ts_id1, exams1 in slot_map.items():
            for ts_id2, exams2 in slot_map.items():
                pair = tuple(sorted([ts_id1, ts_id2]))
                if pair in checked:
                    continue
                checked.add(pair)
                ts1 = self.timeslots[ts_id1]
                ts2 = self.timeslots[ts_id2]
                if not ts1.overlaps(ts2):
                    continue
                for e1 in exams1:
                    for e2 in exams2:
                        if e1 >= e2:
                            continue
                        if e2 in self.conflict_graph.get(e1, set()):
                            shared = len(
                                self.exams[e1].enrolled_students
                                & self.exams[e2].enrolled_students
                            )
                            violations.append(ConstraintViolation(
                                "student_clash", ConstraintType.HARD,
                                self.HARD_PENALTY * shared,
                                f"Exams {e1} & {e2} clash ({shared} students)",
                                exam_ids=[e1, e2],
                            ))
        return violations

    def _check_room_clash(self, schedule):
        violations = []
        room_time = {}
        for a in schedule:
            room_time.setdefault(a.room_id, []).append(a)

        for room_id, assignments in room_time.items():
            for i in range(len(assignments)):
                for j in range(i + 1, len(assignments)):
                    ts_i = self.timeslots[assignments[i].timeslot_id]
                    ts_j = self.timeslots[assignments[j].timeslot_id]
                    if ts_i.overlaps(ts_j):
                        violations.append(ConstraintViolation(
                            "room_clash", ConstraintType.HARD,
                            self.HARD_PENALTY,
                            f"Room {room_id}: exams {assignments[i].exam_id} & {assignments[j].exam_id} overlap",
                            exam_ids=[assignments[i].exam_id, assignments[j].exam_id],
                        ))
        return violations

    def _check_room_capacity(self, schedule):
        violations = []
        for a in schedule:
            exam = self.exams[a.exam_id]
            room = self.rooms[a.room_id]
            if exam.student_count > room.capacity:
                overflow = exam.student_count - room.capacity
                violations.append(ConstraintViolation(
                    "room_capacity", ConstraintType.HARD,
                    self.HARD_PENALTY * overflow,
                    f"Exam {a.exam_id} has {exam.student_count} students but room {a.room_id} seats {room.capacity}",
                    exam_ids=[a.exam_id],
                ))
        return violations

    def _check_exam_room_type_match(self, schedule):
        """
        HC4: Exam type must match room type.
        'hands-on' exams can ONLY go in 'lab' rooms.
        'written' exams can ONLY go in 'lec' rooms.
        This replaces the old _check_computer_requirement method.
        """
        violations = []
        for a in schedule:
            exam = self.exams[a.exam_id]
            room = self.rooms[a.room_id]
            if exam.exam_type == 'hands-on' and room.room_type != 'lab':
                violations.append(ConstraintViolation(
                    "exam_room_type", ConstraintType.HARD,
                    self.HARD_PENALTY,
                    f"Hands-on exam {a.exam_id} placed in non-lab room {a.room_id}",
                    exam_ids=[a.exam_id],
                ))
            elif exam.exam_type == 'written' and room.room_type != 'lec':
                violations.append(ConstraintViolation(
                    "exam_room_type", ConstraintType.HARD,
                    self.HARD_PENALTY,
                    f"Written exam {a.exam_id} placed in lab room {a.room_id}",
                    exam_ids=[a.exam_id],
                ))
        return violations

    def _check_instructor_clash(self, schedule):
        """HC5: No instructor has two exams in overlapping timeslots.
        Exception: sections of the same subject (same group_id) are EXPECTED
        to share a timeslot since one instructor handles all sections."""
        violations = []
        # Build instructor → [(exam_id, timeslot_id)] map from the schedule
        instructor_schedule = {}
        for a in schedule:
            exam = self.exams[a.exam_id]
            if exam.instructor_id:
                iid = str(exam.instructor_id)
                instructor_schedule.setdefault(iid, []).append(a)

        for iid, assignments in instructor_schedule.items():
            for i in range(len(assignments)):
                for j in range(i + 1, len(assignments)):
                    # Skip if both exams belong to the same group (same subject sections)
                    exam_i = self.exams[assignments[i].exam_id]
                    exam_j = self.exams[assignments[j].exam_id]
                    if exam_i.group_id and exam_i.group_id == exam_j.group_id:
                        continue
                    ts_i = self.timeslots[assignments[i].timeslot_id]
                    ts_j = self.timeslots[assignments[j].timeslot_id]
                    if ts_i.overlaps(ts_j):
                        violations.append(ConstraintViolation(
                            "instructor_clash", ConstraintType.HARD,
                            self.HARD_PENALTY,
                            f"Instructor {iid}: exams {assignments[i].exam_id} & {assignments[j].exam_id} overlap",
                            exam_ids=[assignments[i].exam_id, assignments[j].exam_id],
                        ))
        return violations

    def _check_section_same_timeslot(self, schedule):
        """
        HC6: All sections of the same subject (same group_id) MUST share
        the exact same timeslot. This is a hard constraint — if Section A
        of Math 101 is at 9:30 AM Monday, Section B must also be at 9:30 AM Monday.
        """
        violations = []
        # Build exam → timeslot lookup from the schedule
        exam_ts = {a.exam_id: a.timeslot_id for a in schedule}

        for gid, eids in self.group_map.items():
            if len(eids) <= 1:
                continue  # single section, nothing to check
            # Get the timeslot for each exam in this group
            timeslots_used = set()
            for eid in eids:
                if eid in exam_ts:
                    timeslots_used.add(exam_ts[eid])
            if len(timeslots_used) > 1:
                violations.append(ConstraintViolation(
                    "section_same_timeslot", ConstraintType.HARD,
                    self.HARD_PENALTY * (len(timeslots_used) - 1),
                    f"Group {gid}: sections are in {len(timeslots_used)} different timeslots (must be 1)",
                    exam_ids=eids,
                ))
        return violations

    # ── Soft Constraints ──────────────────────────────────────────────────

    def _check_section_different_rooms(self, schedule):
        """
        SC: Sections of the same subject SHOULD be in different rooms.
        If two sections share the same room and timeslot, penalize lightly.
        """
        violations = []
        exam_assignment = {a.exam_id: a for a in schedule}

        for gid, eids in self.group_map.items():
            if len(eids) <= 1:
                continue
            rooms_used = []
            for eid in eids:
                if eid in exam_assignment:
                    rooms_used.append(exam_assignment[eid].room_id)
            # Check for duplicate rooms within same group
            if len(rooms_used) != len(set(rooms_used)):
                duplicates = len(rooms_used) - len(set(rooms_used))
                violations.append(ConstraintViolation(
                    "section_same_room", ConstraintType.SOFT,
                    30 * duplicates,
                    f"Group {gid}: {duplicates} section pair(s) share the same room",
                    exam_ids=eids,
                ))
        return violations

    def _check_consecutive_exams(self, schedule):
        violations = []
        student_schedule = {}
        for a in schedule:
            exam = self.exams[a.exam_id]
            ts = self.timeslots[a.timeslot_id]
            for sid in exam.enrolled_students:
                student_schedule.setdefault(sid, []).append((a.exam_id, ts))

        consecutive_count = 0
        for sid, entries in student_schedule.items():
            entries.sort(key=lambda x: (x[1].day, x[1].start_hour, x[1].start_minute))
            for i in range(len(entries) - 1):
                if entries[i][1].is_consecutive(entries[i + 1][1]):
                    consecutive_count += 1

        if consecutive_count > 0:
            violations.append(ConstraintViolation(
                "consecutive_exams", ConstraintType.SOFT,
                50 * consecutive_count,
                f"{consecutive_count} student-consecutive-exam pairs"
            ))
        return violations

    def _check_spread(self, schedule):
        violations = []
        day_counts = {}
        for a in schedule:
            day = self.timeslots[a.timeslot_id].day
            day_counts[day] = day_counts.get(day, 0) + 1

        if day_counts:
            counts = list(day_counts.values())
            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            penalty = variance * 10
            if penalty > 0:
                violations.append(ConstraintViolation(
                    "uneven_spread", ConstraintType.SOFT, penalty,
                    f"Day distribution variance: {variance:.1f}"
                ))
        return violations

    def _check_room_utilization(self, schedule):
        violations = []
        total_waste = 0
        for a in schedule:
            exam = self.exams[a.exam_id]
            room = self.rooms[a.room_id]
            waste = room.capacity - exam.student_count
            if waste > 0:
                total_waste += waste
        penalty = total_waste * 0.5
        if penalty > 0:
            violations.append(ConstraintViolation(
                "room_waste", ConstraintType.SOFT, penalty,
                f"Total wasted seats: {total_waste}"
            ))
        return violations

    def _check_instructor_preferences(self, schedule):
        """
        SC4: Instructor time preferences (soft constraint).

        If an exam's instructor has set preferred days and/or shifts,
        apply a minor penalty (+20 per mismatch) when the scheduled
        timeslot falls outside those preferences.
        The algorithm will TRY to respect them but will not fail if
        the preferred slots are full.
        """
        violations = []
        penalty_total = 0
        for a in schedule:
            exam = self.exams[a.exam_id]
            prefs = exam.instructor_prefs
            if not prefs:
                continue

            ts = self.timeslots[a.timeslot_id]
            pref_days = prefs.get('days', [])
            pref_shifts = prefs.get('shifts', [])

            # Check day preference
            if pref_days and ts.day not in pref_days:
                penalty_total += 20

            # Check shift preference
            if pref_shifts and ts.shift_name not in pref_shifts:
                penalty_total += 20

        if penalty_total > 0:
            violations.append(ConstraintViolation(
                "instructor_preferences", ConstraintType.SOFT,
                penalty_total,
                f"Instructor preference mismatches: {penalty_total // 20} violations"
            ))
        return violations

    # ── Main Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, schedule):
        all_violations = []

        # Hard
        all_violations.extend(self._check_student_clash(schedule))
        all_violations.extend(self._check_room_clash(schedule))
        all_violations.extend(self._check_room_capacity(schedule))
        all_violations.extend(self._check_exam_room_type_match(schedule))
        all_violations.extend(self._check_instructor_clash(schedule))
        all_violations.extend(self._check_section_same_timeslot(schedule))

        # Soft
        all_violations.extend(self._check_consecutive_exams(schedule))
        all_violations.extend(self._check_spread(schedule))
        all_violations.extend(self._check_room_utilization(schedule))
        all_violations.extend(self._check_instructor_preferences(schedule))
        all_violations.extend(self._check_section_different_rooms(schedule))

        total_penalty = sum(v.penalty for v in all_violations)
        fitness = 10_000_000 - total_penalty

        return fitness, all_violations

    def is_feasible(self, schedule):
        _, violations = self.evaluate(schedule)
        return not any(v.constraint_type == ConstraintType.HARD for v in violations)