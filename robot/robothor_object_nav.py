from flask import Flask, Response, render_template_string, request, jsonify
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from collections import deque
from PIL import Image
import io
import time
import threading
import cv2
import numpy as np

app = Flask(__name__)

GRID = 0.10
PORT = 5000
HOUSE_SPLIT = "train"
HOUSE_INDEX = None
HOUSE_SEARCH_LIMIT = 200
ROOM_NAV_LABELS = [
    "kitchen",
    "bathroom",
    "bedroom",
    "living_room",
    "exercise_room",
]

ROTATE_DELAY = 0.5
MOVE_DELAY = 0.2
REFRESH_MS = 150

PLACE_TO_ROOMS = {
    "kitchen": ["kitchen"],
    "bathroom": ["bathroom"],
    "bedroom": ["bedroom"],
    "living_room": ["living_room", "livingroom"],
    "exercise_room": ["exercise_room", "gym", "fitness_room"],
}


def normalize_label(label):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_room_type(room_type):
    return normalize_label(room_type).replace("_", "")


def score_house_for_places(house_json):
    room_types = {
        normalize_room_type(room.get("roomType", ""))
        for room in house_json.get("rooms", [])
    }

    score = 0
    for label in ROOM_NAV_LABELS:
        aliases = PLACE_TO_ROOMS.get(label, [])

        if any(normalize_room_type(alias) in room_types for alias in aliases):
            score += 1

    return score


def load_procthor_house(split=HOUSE_SPLIT, index=HOUSE_INDEX):
    try:
        import prior
    except ImportError as exc:
        raise RuntimeError(
            "ProcTHOR를 쓰려면 prior 패키지가 필요합니다. "
            "먼저 `pip install prior procthor`를 실행해주세요."
        ) from exc

    dataset = prior.load_dataset("procthor-10k")

    if index is not None:
        return dataset[split][index], index

    best_index = 0
    best_house = dataset[split][0]
    best_score = score_house_for_places(best_house)
    search_limit = min(HOUSE_SEARCH_LIMIT, len(dataset[split]))

    for candidate_index in range(1, search_limit):
        candidate_house = dataset[split][candidate_index]
        candidate_score = score_house_for_places(candidate_house)

        if candidate_score > best_score:
            best_index = candidate_index
            best_house = candidate_house
            best_score = candidate_score

        if best_score == len(ROOM_NAV_LABELS):
            break

    return best_house, best_index


house, house_index = load_procthor_house()

controller = Controller(
    scene=house,
    agentMode="locobot",
    width=800,
    height=600,
    gridSize=GRID,
    rotateStepDegrees=90,
    platform=CloudRendering
)

# room 확인
room_types = sorted(set(room.get("roomType", "Unknown") for room in house["rooms"]))
print("ProcTHOR house:", HOUSE_SPLIT, house_index, flush=True)
print("room types:", room_types, flush=True)

controller_lock = threading.RLock()

nav_status = {
    "running": False,
    "message": "Ready",
    "house": f"{HOUSE_SPLIT} {house_index}",
    "room_types": room_types,
    "supported_places": sorted(PLACE_TO_ROOMS.keys()),
    "step": 0,
    "target_place": None,
    "target_room": None,
    "start": None,
    "goal": None,
    "path_length": None,
}

frame_condition = threading.Condition()
latest_frame = None
frame_id = 0

HTML = """
<!doctype html>
<html>
<head>
    <title>ProcTHOR Place Navigation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        img {
            border: 1px solid #999;
        }
        button {
            height: 36px;
            margin: 3px;
        }
        input {
            height: 30px;
            width: 180px;
            font-size: 16px;
        }
        #status {
            white-space: pre-line;
            padding: 10px;
            border: 1px solid #ccc;
            width: 780px;
            background: #f7f7f7;
        }
    </style>
</head>
<body>
    <h2>ProcTHOR Place Navigation</h2>

    <img id="view" src="/video_feed" width="800">

    <br><br>

    <b>Place label:</b>
    <input id="place" value="kitchen">
    <button onclick="goPlace()">GO</button>

    <br><br>

    <b>Manual control:</b><br>
    <button onclick="sendAction('MoveAhead')">W / MoveAhead</button>
    <button onclick="sendAction('RotateLeft')">A / RotateLeft</button>
    <button onclick="sendAction('MoveBack')">S / MoveBack</button>
    <button onclick="sendAction('RotateRight')">D / RotateRight</button>

    <br><br>

    <div id="status">Loading...</div>

<script>
function refreshImage() {
    // MJPEG stream handles image updates
}

function refreshStatus() {
    fetch("/status")
    .then(r => r.json())
    .then(data => {
        document.getElementById("status").innerText =
            "running: " + data.running + "\\n" +
            "message: " + data.message + "\\n" +
            "house: " + data.house + "\\n" +
            "room_types: " + JSON.stringify(data.room_types) + "\\n" +
            "supported_places: " + JSON.stringify(data.supported_places) + "\\n" +
            "step: " + data.step + "\\n" +
            "target_place: " + data.target_place + "\\n" +
            "target_room: " + data.target_room + "\\n" +
            "start: " + JSON.stringify(data.start) + "\\n" +
            "goal: " + JSON.stringify(data.goal) + "\\n" +
            "path_length: " + data.path_length;
    });
}

function sendAction(action) {
    fetch("/action/" + action, {method: "POST"})
    .then(() => {
        refreshImage();
        refreshStatus();
    });
}

function goPlace() {
    const place = document.getElementById("place").value;

    fetch("/go", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({place: place})
    })
    .then(() => {
        refreshStatus();
    });
}

document.addEventListener("keydown", function(event) {
    const key = event.key.toLowerCase();

    if (key === "w") sendAction("MoveAhead");
    if (key === "a") sendAction("RotateLeft");
    if (key === "s") sendAction("MoveBack");
    if (key === "d") sendAction("RotateRight");
});

setInterval(() => {
    refreshImage();
    refreshStatus();
}, REFRESH_MS_PLACEHOLDER);
</script>
</body>
</html>
""".replace("REFRESH_MS_PLACEHOLDER", str(REFRESH_MS))


def log(msg):
    print(msg, flush=True)


def round_pos(pos):
    return (round(pos["x"], 2), round(pos["z"], 2))


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def controller_step(action, **kwargs):
    with controller_lock:
        event = controller.step(action=action, **kwargs)

    update_latest_frame(event)
    return event


def get_last_event():
    with controller_lock:
        return controller.last_event


def get_reachable_positions():
    event = controller_step(action="GetReachablePositions")
    return [round_pos(p) for p in event.metadata["actionReturn"]]



def polygon_center(points):
    xs = [p["x"] for p in points]
    zs = [p["z"] for p in points]
    return (round(sum(xs) / len(xs), 2), round(sum(zs) / len(zs), 2))


def point_in_polygon(point, polygon):
    x, z = point
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi = polygon[i]["x"]
        zi = polygon[i]["z"]
        xj = polygon[j]["x"]
        zj = polygon[j]["z"]

        intersects = ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-9) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def find_target_room(place_label):
    room_candidates = PLACE_TO_ROOMS.get(place_label, [])
    wanted = {normalize_room_type(room) for room in room_candidates}

    for room in house["rooms"]:
        room_type = normalize_room_type(room.get("roomType", ""))

        if room_type in wanted:
            return room

    return None


def nearest_reachable_position_to_point(point, reachable):
    px, pz = point

    return min(
        reachable,
        key=lambda p: (p[0] - px) ** 2 + (p[1] - pz) ** 2
    )


def nearest_reachable_position_in_room(room, reachable):
    polygon = room["floorPolygon"]
    center = polygon_center(polygon)
    inside_room = [p for p in reachable if point_in_polygon(p, polygon)]

    if inside_room:
        return nearest_reachable_position_to_point(center, inside_room)

    return nearest_reachable_position_to_point(center, reachable)


def resolve_place_goal(place_label, reachable):
    target_room = find_target_room(place_label)

    if target_room is None:
        return None

    goal = nearest_reachable_position_in_room(target_room, reachable)

    return {
        "goal": goal,
        "room": target_room,
        "via": f"room {target_room.get('roomType', 'Unknown')}",
    }


def update_latest_frame(event):
    global latest_frame, frame_id

    rgb = event.frame
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not ok:
        return

    with frame_condition:
        latest_frame = jpeg.tobytes()
        frame_id += 1
        frame_condition.notify_all()


def bfs(start, goal, reachable):
    reachable = set(reachable)

    q = deque([start])
    parent = {start: None}

    moves = [
        (0, GRID),
        (GRID, 0),
        (0, -GRID),
        (-GRID, 0),
    ]

    while q:
        cur = q.popleft()

        if cur == goal:
            break

        for dx, dz in moves:
            nxt = (round(cur[0] + dx, 2), round(cur[1] + dz, 2))

            if nxt in reachable and nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    if goal not in parent:
        return []

    path = []
    cur = goal

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    return path[::-1]


def direction_to_rotation(dx, dz):
    if dz > 0:
        return 0
    if dx > 0:
        return 90
    if dz < 0:
        return 180
    if dx < 0:
        return 270

    return None


def rotate_to(target_rotation):
    event = get_last_event()
    current = round(event.metadata["agent"]["rotation"]["y"]) % 360

    max_turns = 8
    turns = 0

    while angle_diff(current, target_rotation) > 12 and turns < max_turns:
        diff = (target_rotation - current) % 360

        if diff <= 180:
            action = "RotateRight"
        else:
            action = "RotateLeft"

        event = controller_step(action=action)
        current = round(event.metadata["agent"]["rotation"]["y"]) % 360

        nav_status["step"] += 1
        nav_status["message"] = f"{action} → rotation {current}"

        log(f"{action} True rotation={current}")

        turns += 1
        time.sleep(ROTATE_DELAY)

    if turns >= max_turns:
        log(f"Rotation warning: target={target_rotation}, current={current}")


def follow_path(path):
    for i in range(len(path) - 1):
        x1, z1 = path[i]
        x2, z2 = path[i + 1]

        dx = round(x2 - x1, 2)
        dz = round(z2 - z1, 2)

        target_rotation = direction_to_rotation(dx, dz)

        if target_rotation is None:
            log(f"Skip invalid movement: {(x1, z1)} -> {(x2, z2)}")
            continue

        rotate_to(target_rotation)

        event = controller_step(action="MoveAhead")

        pos = event.metadata["agent"]["position"]
        success = event.metadata["lastActionSuccess"]

        nav_status["step"] += 1
        nav_status["message"] = f"MoveAhead {success} position={pos}"

        log(f"MoveAhead {success} {pos}")

        time.sleep(MOVE_DELAY)


def go_to_place_worker(place_label):
    nav_status["running"] = True
    nav_status["message"] = "Planning"
    nav_status["step"] = 0
    place_label = normalize_label(place_label)

    nav_status["target_place"] = place_label
    nav_status["target_room"] = None
    nav_status["start"] = None
    nav_status["goal"] = None
    nav_status["path_length"] = None

    log("")
    log("========== Navigation Start ==========")
    log(f"place: {place_label}")

    try:
        event = get_last_event()

        start = round_pos(event.metadata["agent"]["position"])
        reachable = get_reachable_positions()

        target = resolve_place_goal(place_label, reachable)

        if target is None:
            msg = f"No room found for place: {place_label}"
            nav_status["message"] = msg
            log(msg)
            nav_status["running"] = False
            return

        goal = target["goal"]
        target_room = target["room"]
        path = bfs(start, goal, reachable)

        nav_status["target_room"] = target_room.get("roomType", "Unknown")

        nav_status["start"] = start
        nav_status["goal"] = goal
        nav_status["path_length"] = len(path)

        log(f"target via: {target['via']}")
        log(f"start: {start}")
        log(f"goal: {goal}")
        log(f"path length: {len(path)}")
        log(f"path: {path}")

        if not path:
            msg = "No path found"
            nav_status["message"] = msg
            log(msg)
            nav_status["running"] = False
            return

        nav_status["message"] = f"Going to {place_label} via {target['via']}"

        follow_path(path)

        msg = f"Arrived at: {place_label} ({target['via']})"
        nav_status["message"] = msg
        log(msg)
        log("========== Navigation End ==========")

    except Exception as e:
        msg = f"Error: {e}"
        nav_status["message"] = msg
        log(msg)

    nav_status["running"] = False


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/image")
def image():
    event = get_last_event()
    img = Image.fromarray(event.frame)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(buf.getvalue(), mimetype="image/png")

@app.route("/video_feed")
def video_feed():
    def generate():
        last_sent_id = -1

        while True:
            with frame_condition:
                frame_condition.wait_for(
                    lambda: latest_frame is not None and frame_id != last_sent_id
                )

                frame = latest_frame
                last_sent_id = frame_id

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/status")
def status():
    return jsonify(nav_status)


@app.route("/action/<action>", methods=["POST"])
def manual_action(action):
    if nav_status["running"]:
        return "navigation running"

    event = controller_step(action=action)

    pos = event.metadata["agent"]["position"]
    success = event.metadata["lastActionSuccess"]

    nav_status["step"] += 1
    nav_status["message"] = f"Manual {action} {success} position={pos}"

    log(f"Manual {action} {success} {pos}")

    return "ok"


@app.route("/go", methods=["POST"])
def go():
    if nav_status["running"]:
        return "already running"

    data = request.get_json()
    place = data.get("place", "kitchen").strip()

    thread = threading.Thread(
        target=go_to_place_worker,
        args=(place,),
        daemon=True
    )
    thread.start()

    return "started"


if __name__ == "__main__":
    update_latest_frame(controller.last_event)
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=PORT,
            use_reloader=False,
            threaded=True
        ),
        daemon=True
    )

    flask_thread.start()

    log(f"Open browser: http://localhost:{PORT}")
    log("Browser UI is ready.")
    log("Type a place label in terminal or use the browser GO button.")
    log("Type quit or exit to stop terminal input loop.")

    while True:
        place = input(
            "\nPlace label ex) kitchen, bathroom, bedroom, living_room, exercise_room: "
        ).strip()

        if place.lower() in ["quit", "exit"]:
            break

        if not place:
            continue

        if nav_status["running"]:
            print("Navigation already running.")
            continue

        thread = threading.Thread(
            target=go_to_place_worker,
            args=(place,),
            daemon=True
        )
        thread.start()
