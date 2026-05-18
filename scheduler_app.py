"""
Flask API for Hybrid Genetic Exam Scheduler
============================================
Updated to work with Laravel exam scheduling system.

Endpoints:
  POST /api/schedule/sync   → Run scheduler synchronously (used by Laravel)
  POST /api/schedule        → Run scheduler async (returns job_id)
  GET  /api/schedule/<id>   → Get async result
  GET  /api/health          → Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import threading
import time
from datetime import datetime

from scheduler_constraints import Exam, Room, Timeslot, ConstraintEngine, ConstraintType
from scheduler_genetic_algorithm import ExamSchedulerGA, GAConfig, format_schedule

app = Flask(__name__)
CORS(app)

results_store: dict = {}
jobs_store: dict = {}


# ─── Request Parsing ──────────────────────────────────────────────────────────

def parse_exams(data: list) -> list:
    """Parse exam data from Laravel format."""
    exams = []
    for e in data:
        exams.append(Exam(
            id=str(e["id"]),
            name=e.get("name", str(e["id"])),
            duration_minutes=e.get("duration_minutes", 120),
            student_count=e.get("student_count", 0),
            enrolled_students=set(str(s) for s in e.get("enrolled_students", [])),
            department=e.get("department", ""),
            requires_computer=e.get("requires_computer", False),
            priority=e.get("priority", 0),
            instructor_id=e.get("instructor_id", None),
            exam_type=e.get("exam_type", "written"),
            instructor_prefs=e.get("instructor_prefs", {}),
            group_id=e.get("group_id", ""),
            section_id=str(e.get("section_id", "")),
        ))
    return exams


def parse_rooms(data: list) -> list:
    rooms = []
    for r in data:
        # Determine room_type: use explicit value, or fall back to has_computers flag
        room_type = r.get("room_type", "lab" if r.get("has_computers", False) else "lec")
        rooms.append(Room(
            id=str(r["id"]),
            name=r.get("name", str(r["id"])),
            capacity=r.get("capacity", 50),
            has_computers=r.get("has_computers", False),
            building=r.get("building", ""),
            room_type=room_type,
        ))
    return rooms


def parse_timeslots(data: list) -> list:
    slots = []
    for t in data:
        slots.append(Timeslot(
            id=str(t["id"]),
            day=t["day"],
            start_hour=t["start_hour"],
            start_minute=t.get("start_minute", 0),
            duration_minutes=t.get("duration_minutes", 120),
            # Carry the actual date string for Laravel
            date_str=t.get("date_str", ""),
        ))
    return slots


# ─── Enhanced format_schedule that includes Laravel-needed fields ─────────────

def format_schedule_laravel(schedule, exams_dict, rooms_dict, slots_dict):
    """Format schedule with all fields Laravel needs to create exam_schedule records."""
    result = []
    for a in sorted(schedule, key=lambda x: (
        slots_dict[x.timeslot_id].day,
        slots_dict[x.timeslot_id].start_hour,
        slots_dict[x.timeslot_id].start_minute,
    )):
        ts = slots_dict[a.timeslot_id]
        room = rooms_dict[a.room_id]
        exam = exams_dict[a.exam_id]

        result.append({
            # IDs for Laravel to save — subject_id is the composite "5_12"
            # Laravel post-processes this to extract real subject_id and section_id
            "subject_id": a.exam_id,
            "room_id": int(a.room_id),
            "instructor_id": getattr(exam, 'instructor_id', None),

            # Schedule details
            "exam_date": ts.date_str if hasattr(ts, 'date_str') and ts.date_str else f"day_{ts.day}",
            "start_time": f"{ts.start_hour:02d}:{ts.start_minute:02d}",
            "end_time_computed": _compute_end_time(ts.start_hour, ts.start_minute, exam.duration_minutes),
            "duration_minutes": exam.duration_minutes,

            # Display info
            "exam_name": exam.name,
            "department": exam.department,
            "student_count": exam.student_count,
            "day_name": ts.day_name,
            "room_name": room.name,
            "room_capacity": room.capacity,
            "building": getattr(room, 'building', ''),
        })
    return result


def _compute_end_time(start_hour, start_minute, duration_minutes):
    """Compute end time string from start + duration."""
    total_minutes = start_hour * 60 + start_minute + duration_minutes
    end_hour = (total_minutes // 60) % 24
    end_minute = total_minutes % 60
    return f"{end_hour:02d}:{end_minute:02d}"


# ─── Background Runner ────────────────────────────────────────────────────────

def run_scheduler_job(job_id, exams, rooms, timeslots, config):
    jobs_store[job_id]["status"] = "running"
    jobs_store[job_id]["started_at"] = datetime.utcnow().isoformat()

    generation_log = []

    def on_generation(stats):
        generation_log.append({
            "generation": stats.generation,
            "best_fitness": stats.best_fitness,
            "hard_violations": stats.hard_violations,
            "feasible_pct": stats.feasible_pct,
            "elapsed_sec": round(stats.elapsed_sec, 2),
        })
        jobs_store[job_id]["progress"] = {
            "current_generation": stats.generation,
            "max_generations": config.max_generations,
            "best_fitness": stats.best_fitness,
            "feasible_pct": stats.feasible_pct,
        }

    try:
        ga = ExamSchedulerGA(exams, rooms, timeslots, config)
        ga.initialize_population()
        best_schedule, best_fitness, history = ga.run(callback=on_generation)

        exams_dict = {e.id: e for e in exams}
        rooms_dict = {r.id: r for r in rooms}
        slots_dict = {t.id: t for t in timeslots}

        engine = ConstraintEngine(exams_dict, rooms_dict, slots_dict)
        _, violations = engine.evaluate(best_schedule)

        result = {
            "job_id": job_id,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "fitness": best_fitness,
            "is_feasible": engine.is_feasible(best_schedule),
            "schedule": format_schedule_laravel(best_schedule, exams_dict, rooms_dict, slots_dict),
            "violations": [
                {"name": v.constraint_name, "type": v.constraint_type.value,
                 "penalty": v.penalty, "description": v.description}
                for v in violations
            ],
            "summary": {
                "total_exams": len(exams),
                "total_rooms": len(rooms),
                "total_timeslots": len(timeslots),
                "generations_run": len(generation_log),
                "hard_violations": sum(1 for v in violations if v.constraint_type == ConstraintType.HARD),
                "soft_violations": sum(1 for v in violations if v.constraint_type == ConstraintType.SOFT),
            },
            "evolution_log": generation_log[-10:],  # Last 10 generations only
        }

        results_store[job_id] = result
        jobs_store[job_id]["status"] = "completed"

    except Exception as e:
        import traceback
        jobs_store[job_id]["status"] = "failed"
        jobs_store[job_id]["error"] = str(e)
        jobs_store[job_id]["traceback"] = traceback.format_exc()
        results_store[job_id] = {"job_id": job_id, "status": "failed", "error": str(e)}


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def root():
    """Helps verify the correct app is bound to the port (avoid silent wrong-process 404s)."""
    return jsonify({
        "service": "exam-scheduler-ga",
        "status": "ok",
        "endpoints": {
            "GET /api/health": "liveness",
            "POST /api/schedule/sync": "synchronous schedule (Laravel)",
            "POST /api/schedule": "async schedule",
        },
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "exam-scheduler-ga"})


@app.route("/api/schedule", methods=["POST"])
def create_schedule():
    """Async scheduling — returns job_id to poll."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    for field in ["exams", "rooms", "timeslots"]:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        exams = parse_exams(data["exams"])
        rooms = parse_rooms(data["rooms"])
        timeslots = parse_timeslots(data["timeslots"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

    if not exams or not rooms or not timeslots:
        return jsonify({"error": "Need at least 1 exam, 1 room, and 1 timeslot"}), 400

    cfg_data = data.get("config", {})
    config = GAConfig(
        population_size=cfg_data.get("population_size", 100),
        max_generations=cfg_data.get("max_generations", 300),
        mutation_rate=cfg_data.get("mutation_rate", 0.15),
        stagnation_limit=cfg_data.get("stagnation_limit", 50),
        seed=cfg_data.get("seed", None),
        soft_weight=float(cfg_data.get("soft_weight", 1.0)),
        max_repair_attempts=int(cfg_data.get("max_repair_attempts", 50)),
        parallel_eval_workers=int(cfg_data.get("parallel_eval_workers", 0)),
        early_exit_feasible_stagnation=int(cfg_data.get("early_exit_feasible_stagnation", 0)),
        local_search_steps=int(cfg_data.get("local_search_steps", 20)),
        local_search_ratio=float(cfg_data.get("local_search_ratio", 0.05)),
    )

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id, "status": "queued",
        "created_at": datetime.utcnow().isoformat(), "progress": None,
    }

    thread = threading.Thread(
        target=run_scheduler_job,
        args=(job_id, exams, rooms, timeslots, config), daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/schedule/<job_id>", methods=["GET"])
def get_schedule(job_id):
    if job_id in results_store:
        return jsonify(results_store[job_id])
    if job_id in jobs_store:
        return jsonify(jobs_store[job_id])
    return jsonify({"error": "Job not found"}), 404


@app.route("/api/schedule/sync", methods=["POST"])
def create_schedule_sync():
    """Synchronous — blocks until complete. Used by Laravel."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    for field in ["exams", "rooms", "timeslots"]:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        exams = parse_exams(data["exams"])
        rooms = parse_rooms(data["rooms"])
        timeslots = parse_timeslots(data["timeslots"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

    if not exams or not rooms or not timeslots:
        return jsonify({"error": "Need at least 1 exam, 1 room, and 1 timeslot"}), 400

    cfg_data = data.get("config", {})
    config = GAConfig(
        population_size=cfg_data.get("population_size", 80),
        max_generations=cfg_data.get("max_generations", 200),
        mutation_rate=cfg_data.get("mutation_rate", 0.15),
        stagnation_limit=cfg_data.get("stagnation_limit", 40),
        seed=cfg_data.get("seed", None),
        soft_weight=float(cfg_data.get("soft_weight", 1.0)),
        max_repair_attempts=int(cfg_data.get("max_repair_attempts", 50)),
        parallel_eval_workers=int(cfg_data.get("parallel_eval_workers", 0)),
        early_exit_feasible_stagnation=int(cfg_data.get("early_exit_feasible_stagnation", 0)),
        local_search_steps=int(cfg_data.get("local_search_steps", 20)),
        local_search_ratio=float(cfg_data.get("local_search_ratio", 0.05)),
    )

    try:
        ga = ExamSchedulerGA(exams, rooms, timeslots, config)
        ga.initialize_population()
        best_schedule, best_fitness, history = ga.run()

        exams_dict = {e.id: e for e in exams}
        rooms_dict = {r.id: r for r in rooms}
        slots_dict = {t.id: t for t in timeslots}

        engine = ConstraintEngine(exams_dict, rooms_dict, slots_dict)
        _, violations = engine.evaluate(best_schedule)

        return jsonify({
            "status": "completed",
            "fitness": best_fitness,
            "is_feasible": engine.is_feasible(best_schedule),
            "schedule": format_schedule_laravel(best_schedule, exams_dict, rooms_dict, slots_dict),
            "violations": [
                {"name": v.constraint_name, "type": v.constraint_type.value,
                 "penalty": v.penalty, "description": v.description}
                for v in violations
            ],
            "summary": {
                "total_exams": len(exams),
                "total_rooms": len(rooms),
                "total_timeslots": len(timeslots),
                "generations_run": len(history),
                "hard_violations": sum(1 for v in violations if v.constraint_type == ConstraintType.HARD),
                "soft_violations": sum(1 for v in violations if v.constraint_type == ConstraintType.SOFT),
            },
        })
    except Exception as e:
        import traceback
        return jsonify({"status": "failed", "error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    # Default: single process (use_reloader=False). Debug mode with reloader=True spawns an extra
    # Python parent process — netstat often shows two PIDs on :5000 and restarts can briefly serve stale code.
    # Enable hot-reload: set SCHEDULER_USE_RELOADER=1 in the environment.
    import os

    _reload = os.environ.get("SCHEDULER_USE_RELOADER", "").strip().lower() in ("1", "true", "yes", "on")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=_reload)