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

    # High soft penalty: strongly discourage wrong room type, but allow it
    # as a last resort when no correct-type room is available.
    ROOM_TYPE_MISMATCH_PENALTY = 500_000

    def _check_exam_room_type_match(self, schedule):
        """
        SC (high-penalty soft): Exam type should match room type.

        - 'hands-on' exams strongly prefer 'lab' rooms.
        - 'written'   exams strongly prefer 'lec' rooms.

        This is a SOFT constraint so the algorithm can still produce a
        valid (feasible) schedule when the ideal room type is unavailable,
        rather than generating an unresolvable hard failure.  The penalty
        is set high enough (500,000) that the GA will virtually never place
        an exam in the wrong room type unless there is genuinely no alternative.

        Both directions are penalised equally and flagged with the same
        violation name so the UI can display them clearly.
        """
        violations = []
        for a in schedule:
            exam = self.exams[a.exam_id]
            room = self.rooms[a.room_id]

            if exam.exam_type == 'hands-on' and room.room_type != 'lab':
                violations.append(ConstraintViolation(
                    "room_type_mismatch", ConstraintType.SOFT,
                    self.ROOM_TYPE_MISMATCH_PENALTY,
                    f"Hands-on exam \"{exam.name}\" is in lecture room {room.name} "
                    f"(no lab was available — this is a fallback placement)",
                    exam_ids=[a.exam_id],
                ))
            elif exam.exam_type == 'written' and room.room_type != 'lec':
                violations.append(ConstraintViolation(
                    "room_type_mismatch", ConstraintType.SOFT,
                    self.ROOM_TYPE_MISMATCH_PENALTY,
                    f"Written exam \"{exam.name}\" is in lab room {room.name} "
                    f"(no lecture room was available — this is a fallback placement)",
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
        """Emit one violation per student who has back-to-back exams."""
        violations = []
        student_schedule = {}
        for a in schedule:
            exam = self.exams[a.exam_id]
            ts = self.timeslots[a.timeslot_id]
            for sid in exam.enrolled_students:
                student_schedule.setdefault(sid, []).append((a.exam_id, exam.name, ts))

        for sid, entries in student_schedule.items():
            entries.sort(key=lambda x: (x[2].day, x[2].start_hour, x[2].start_minute))
            for i in range(len(entries) - 1):
                if entries[i][2].is_consecutive(entries[i + 1][2]):
                    e1, e2 = entries[i][1], entries[i + 1][1]
                    violations.append(ConstraintViolation(
                        "consecutive_exams_detail", ConstraintType.SOFT,
                        50,
                        f"Student #{sid} has back-to-back exams: {e1} then {e2} with no break",
                    ))
        return violations

    def _check_spread(self, schedule):
        """Emit one violation per day that is overloaded (above average exam count)."""
        DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        violations = []
        day_counts  = {}
        day_exams   = {}   # day -> [exam_name]
        for a in schedule:
            day  = self.timeslots[a.timeslot_id].day
            name = self.exams[a.exam_id].name
            day_counts[day] = day_counts.get(day, 0) + 1
            day_exams.setdefault(day, []).append(name)

        if not day_counts:
            return violations

        counts = list(day_counts.values())
        mean   = sum(counts) / len(counts)

        for day, count in day_counts.items():
            if count > mean * 1.5 and count > 2:   # only flag notably overloaded days
                day_name  = DAY_NAMES[day] if day < len(DAY_NAMES) else f"Day {day}"
                exams_str = ", ".join(day_exams[day])
                violations.append(ConstraintViolation(
                    "uneven_spread_detail", ConstraintType.SOFT,
                    (count - mean) * 10,
                    f"{day_name} is overloaded with {count} exams (avg {mean:.1f}): {exams_str}",
                ))
        return violations

    def _check_room_utilization(self, schedule):
        """
        Emit one violation per exam assignment whose room has significant waste
        (>= 5 empty seats).  This gives the admin a per-room, per-exam breakdown
        rather than an opaque total.
        """
        violations = []
        for a in schedule:
            exam = self.exams[a.exam_id]
            room = self.rooms[a.room_id]
            waste = room.capacity - exam.student_count
            if waste >= 5:
                violations.append(ConstraintViolation(
                    "room_waste_detail", ConstraintType.SOFT,
                    waste * 0.5,
                    f"Room {room.name} (#{room.id}, cap {room.capacity}): "
                    f"{exam.student_count} student(s) — {waste} seat(s) wasted "
                    f"[{exam.name}]",
                    exam_ids=[a.exam_id],
                ))
        return violations

    def _check_instructor_preferences(self, schedule):
        """
        SC4: Instructor time preferences (soft constraint).

        Emits one ConstraintViolation per instructor per mismatch type
        (day mismatch OR shift mismatch) so the frontend can display a
        detailed, human-readable breakdown of which instructor preference
        was violated and which exam caused it.

        Penalty: +20 per individual mismatch.
        """
        DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # instructor_id -> { 'day': [...], 'shift': [...] }
        per_instructor: dict = {}

        for a in schedule:
            exam = self.exams[a.exam_id]
            prefs = exam.instructor_prefs
            if not prefs:
                continue

            iid = str(exam.instructor_id) if exam.instructor_id else None
            if not iid:
                continue

            ts = self.timeslots[a.timeslot_id]
            # pref_days may arrive as strings ("0","1"…) or ints — normalise to int
            raw_days    = prefs.get('days', [])
            pref_days   = [int(d) for d in raw_days if str(d).lstrip('-').isdigit()]
            pref_shifts = prefs.get('shifts', [])  # list of str like 'morning'

            bucket = per_instructor.setdefault(iid, {'day': [], 'shift': []})

            if pref_days and ts.day not in pref_days:
                pref_day_names = [DAY_NAMES[d] for d in pref_days if 0 <= d < len(DAY_NAMES)]
                bucket['day'].append({
                    'exam_name': exam.name,
                    'scheduled_day': DAY_NAMES[ts.day] if 0 <= ts.day < len(DAY_NAMES) else str(ts.day),
                    'preferred_days': pref_day_names,
                })

            if pref_shifts and ts.shift_name not in pref_shifts:
                bucket['shift'].append({
                    'exam_name': exam.name,
                    'scheduled_shift': ts.shift_name,
                    'preferred_shifts': pref_shifts,
                })

        violations = []
        for iid, mismatches in per_instructor.items():
            if mismatches['day']:
                count  = len(mismatches['day'])
                exams  = ', '.join(m['exam_name'] for m in mismatches['day'])
                sched  = ', '.join(m['scheduled_day'] for m in mismatches['day'])
                prefs  = '/'.join(mismatches['day'][0]['preferred_days'])
                violations.append(ConstraintViolation(
                    "instructor_preference_day", ConstraintType.SOFT,
                    20 * count,
                    f"Instructor {iid} day preference: prefers {prefs} but scheduled on {sched} ({exams})",
                ))

            if mismatches['shift']:
                count  = len(mismatches['shift'])
                exams  = ', '.join(m['exam_name'] for m in mismatches['shift'])
                sched  = ', '.join(m['scheduled_shift'] for m in mismatches['shift'])
                prefs  = '/'.join(mismatches['shift'][0]['preferred_shifts'])
                violations.append(ConstraintViolation(
                    "instructor_preference_shift", ConstraintType.SOFT,
                    20 * count,
                    f"Instructor {iid} shift preference: prefers {prefs} but scheduled in {sched} ({exams})",
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