"""
Benchmark Data: CSUCC BSIT Faculty Loading 2nd Sem SY 2025-2026
================================================================
Encodes the real exam scheduling scenario:
  - 6 comlab rooms, 25 students max each
  - Thursday & Friday only
  - 7:30 AM to 7:30 PM (6 timeslots per day)
"""

from scheduler_constraints import Exam, Room, Timeslot


# ── Student ID generators (synthetic, same section = same students) ───────────
def _students(prefix, count):
    return {f"{prefix}_{i:03d}" for i in range(1, count + 1)}

SEC_STUDENTS = {
    "IA":   _students("Y1A", 25), "IB":   _students("Y1B", 25), "IC":   _students("Y1C", 25),
    "IIA":  _students("Y2A", 20), "IIB":  _students("Y2B", 20), "IIC":  _students("Y2C", 20),
    "IIIA": _students("Y3A", 20), "IIIB": _students("Y3B", 20), "IIIC": _students("Y3C", 20),
    "IVA":  _students("Y4A", 20), "IVB":  _students("Y4B", 20),
}

# ── Instructor IDs ────────────────────────────────────────────────────────────
INST = {
    "Sevilla":    1, "Cuarez":     2, "Grino":     3, "Bojocan":   4,
    "Vistal":     5, "Timbal":     6, "Caday":     7, "Palima":    8,
    "Hempesao":   9, "Escauso":   10, "Lecturer1": 11, "Lecturer2": 12,
    "Anunciado": 13, "Soria":     14, "Castillo":  15,
}


def build_exams():
    """Build all exam objects from the BSIT faculty loading."""
    raw = [
        # (exam_id, name, section_key, student_count, instructor, group_id)
        # ── Sevilla ──
        ("IT104_IA",  "IT 104 Networking 1 (I-A)",       "IA",  25, "Sevilla", "IT104_S"),
        ("IT104_IB",  "IT 104 Networking 1 (I-B)",       "IB",  25, "Sevilla", "IT104_S"),
        ("IT104_IC",  "IT 104 Networking 1 (I-C)",       "IC",  25, "Sevilla", "IT104_S"),
        ("ITE16_IA",  "ITE 16 Info Management (I-A)",    "IA",  25, "Sevilla", "ITE16_Sev"),
        ("IT114_IVA", "IT 114 Adv HCI (IV-A)",           "IVA", 20, "Sevilla", "IT114_Sev"),
        ("IT114_IVB", "IT 114 Adv HCI (IV-B)",           "IVB", 20, "Sevilla", "IT114_Sev"),
        # ── Cuarez ──
        ("IT112_IIIA","IT 112 Sys Admin (III-A)",        "IIIA",20, "Cuarez",  "IT112_C"),
        ("IT112_IIIB","IT 112 Sys Admin (III-B)",        "IIIB",20, "Cuarez",  "IT112_C"),
        ("IT112_IIIC","IT 112 Sys Admin (III-C)",        "IIIC",20, "Cuarez",  "IT112_C"),
        # ── Griño ──
        ("IT113_IVA", "IT 113 Info Assurance 2 (IV-A)",  "IVA", 17, "Grino",   "IT113_G"),
        ("IT113_IVB", "IT 113 Info Assurance 2 (IV-B)",  "IVB", 25, "Grino",   "IT113_G"),
        # ── Bojocan ──
        ("ITE14_IIA", "ITE 14 Data Structures (II-A)",   "IIA", 20, "Bojocan", "ITE14_B"),
        ("ITE14_IIB", "ITE 14 Data Structures (II-B)",   "IIB", 20, "Bojocan", "ITE14_B"),
        ("ITE14_IIC", "ITE 14 Data Structures (II-C)",   "IIC", 20, "Bojocan", "ITE14_B"),
        ("ITE19_IVA", "ITE 19 Competency Appraisal (IV-A)","IVA",20,"Bojocan","ITE19_B"),
        ("ITE19_IVB", "ITE 19 Competency Appraisal (IV-B)","IVB",20,"Bojocan","ITE19_B"),
        # ── Vistal ──
        ("PROF4_IVA", "PROFEL 4 Prof Elective 4 (IV-A)", "IVA", 25, "Vistal",  "PROF4_V"),
        ("PROF4_IVB", "PROFEL 4 Prof Elective 4 (IV-B)", "IVB", 25, "Vistal",  "PROF4_V"),
        ("PROF3V_IVA","PROFEL 3 Prof Elective 3 (IV-A)", "IVA", 25, "Vistal",  "PROF3_Vis"),
        # ── Timbal ──
        ("IT106_IIA", "IT 106 Integrative Prog (II-A)",  "IIA", 25, "Timbal",  "IT106_T"),
        ("IT106_IIB", "IT 106 Integrative Prog (II-B)",  "IIB", 25, "Timbal",  "IT106_T"),
        ("IT106_IIC", "IT 106 Integrative Prog (II-C)",  "IIC", 25, "Timbal",  "IT106_T"),
        # ── Caday ──
        ("CSC104_IIA","CSC 104 OOP (II-A)",              "IIA", 20, "Caday",   "CSC104_C"),
        ("CSC104_IIB","CSC 104 OOP (II-B)",              "IIB", 20, "Caday",   "CSC104_C"),
        ("CSC104_IIC","CSC 104 OOP (II-C)",              "IIC", 20, "Caday",   "CSC104_C"),
        ("PROF2_IIIB","PROFEL 2 Prof Elective 2 (III-B)","IIIB",25, "Caday",   "PROF2_Cad"),
        # ── Palima ──
        ("IT198_IIIA","IT 198 Capstone 2 (III-A)",       "IIIA",20, "Palima",  "IT198_P"),
        ("IT198_IIIB","IT 198 Capstone 2 (III-B)",       "IIIB",20, "Palima",  "IT198_P"),
        ("IT198_IIIC","IT 198 Capstone 2 (III-C)",       "IIIC",20, "Palima",  "IT198_P"),
        ("IT199_IVA", "IT 199 Capstone 1 (IV-A)",        "IVA", 20, "Palima",  "IT199_P"),
        ("IT199_IVB", "IT 199 Capstone 1 (IV-B)",        "IVB", 20, "Palima",  "IT199_P"),
        # ── Hempesao ──
        ("ITE18H_IIA","ITE 18 App Dev (II-A)",           "IIA", 20, "Hempesao","ITE18_H"),
        ("ITE18H_IIB","ITE 18 App Dev (II-B)",           "IIB", 20, "Hempesao","ITE18_H"),
        ("ITE18H_IIC","ITE 18 App Dev (II-C)",           "IIC", 20, "Hempesao","ITE18_H"),
        ("ITE16_IC",  "ITE 16 Info Management (I-C)",    "IC",  25, "Hempesao","ITE16_Hem"),
        # ── Escauso ──
        ("PROF3E_IVB","PROFEL 3 Prof Elective 3 (IV-B)", "IVB", 25, "Escauso", "PROF3_Esc4"),
        ("PROF3E_IIIA","PROFEL 3 Prof Elective 3 (III-A)","IIIA",20,"Escauso","PROF3_Esc3"),
        ("PROF3E_IIIB","PROFEL 3 Prof Elective 3 (III-B)","IIIB",20,"Escauso","PROF3_Esc3"),
        ("PROF3E_IIIC","PROFEL 3 Prof Elective 3 (III-C)","IIIC",20,"Escauso","PROF3_Esc3"),
        # ── Lecturer 1 ──
        ("ITE18L_IVA","ITE 18 App Dev (IV-A)",           "IVA", 20, "Lecturer1","ITE18_L1"),
        ("ITE18L_IVB","ITE 18 App Dev (IV-B)",           "IVB", 20, "Lecturer1","ITE18_L1"),
        ("ITE16L_IIA","ITE 16 Info Mgmt (II-A)",         "IIA", 20, "Lecturer1","ITE16_L1"),
        ("ITE16L_IIB","ITE 16 Info Mgmt (II-B)",         "IIB", 20, "Lecturer1","ITE16_L1"),
        # ── Lecturer 2 ──
        ("IT101_IIIA","IT 101 HCI (III-A)",              "IIIA",20, "Lecturer2","IT101_L2"),
        ("IT101_IIIB","IT 101 HCI (III-B)",              "IIIB",20, "Lecturer2","IT101_L2"),
        ("IT101_IIIC","IT 101 HCI (III-C)",              "IIIC",20, "Lecturer2","IT101_L2"),
        ("ITE13_IB",  "ITE 13 Intermed Prog (I-B)",      "IB",  25, "Lecturer2","ITE13_L2"),
        # ── Anunciado ──
        ("PROF2A_IIIA","PROFEL 2 Embedded Sys (III-A)",  "IIIA",25, "Anunciado","PROF2_Ann"),
        ("PROF2A_IIIC","PROFEL 2 Embedded Sys (III-C)",  "IIIC",25, "Anunciado","PROF2_Ann"),
        # ── Soria ──
        ("IT111_IIIA","IT 111 Platform Tech (III-A)",    "IIIA",20, "Soria",   "IT111_S"),
        ("IT111_IIIB","IT 111 Platform Tech (III-B)",    "IIIB",20, "Soria",   "IT111_S"),
        ("IT111_IIIC","IT 111 Platform Tech (III-C)",    "IIIC",20, "Soria",   "IT111_S"),
        # ── Castillo ──
        ("ITE13_IA",  "ITE 13 Intermed Prog (I-A)",      "IA",  25, "Castillo","ITE13_Cas"),
        ("ITE13_IC",  "ITE 13 Intermed Prog (I-C)",      "IC",  25, "Castillo","ITE13_Cas"),
    ]

    exams = []
    for eid, name, sec_key, count, inst_name, gid in raw:
        exams.append(Exam(
            id=eid,
            name=name,
            duration_minutes=120,
            student_count=count,
            enrolled_students=SEC_STUDENTS[sec_key].copy(),
            exam_type="hands-on",
            instructor_id=INST[inst_name],
            group_id=gid,
            section_id=sec_key,
        ))
    return exams


def build_rooms():
    """6 computer laboratory rooms, 25 capacity each."""
    return [
        Room(id=str(i), name=f"ComLab {i}", capacity=25, room_type="lab")
        for i in range(1, 7)
    ]


def build_timeslots():
    """Thursday & Friday, 6 slots each (7:30 AM to 5:30 PM start), 2-hour exams."""
    slots = []
    idx = 0
    for day_offset, day_num, date_str in [(0, 3, "2026-03-05"), (1, 4, "2026-03-06")]:
        for hour, minute in [(7,30),(9,30),(11,30),(13,30),(15,30),(17,30)]:
            slots.append(Timeslot(
                id=f"TS_{idx}",
                day=day_num,
                start_hour=hour,
                start_minute=minute,
                duration_minutes=120,
                date_str=date_str,
            ))
            idx += 1
    return slots


def get_scenario_description():
    return (
        "CSUCC BSIT Exam Scheduling — 2nd Semester SY 2025-2026\n"
        "College of Engineering and Information Technology\n\n"
        "Scenario Constraints:\n"
        "  • 6 Computer Laboratory rooms (25 students max each)\n"
        "  • Exam days: Thursday and Friday only\n"
        "  • Duty hours: 7:30 AM – 7:30 PM (6 × 2-hour timeslots per day)\n"
        "  • Sections of the same subject must be scheduled at the same time in different rooms\n"
        "  • All exams are hands-on (computer lab required)\n"
        f"  • Total exams to schedule: {len(build_exams())}\n"
        f"  • Total available slots: 12 timeslots × 6 rooms = 72\n"
    )
