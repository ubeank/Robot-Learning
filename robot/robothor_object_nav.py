from flask import Flask, Response, render_template_string, request, jsonify
from ai2thor.controller import Controller
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

ROTATE_DELAY = 0.5
MOVE_DELAY = 0.2
REFRESH_MS = 150

PLACE_TO_OBJECTS = {
    "kitchen": ["Mug", "Sink", "Fridge", "Microwave"],
    "bathroom": ["Sink", "Toilet"],
    "bedroom": ["Bed"],
    "living_room": ["Sofa", "Television"],
}

controller = Controller(
    scene="FloorPlan_Train1_1",
    agentMode="locobot",
    width=800,
    height=600,
    gridSize=GRID,
    rotateStepDegrees=90
)

#object 확인
objects = controller.last_event.metadata["objects"] 
object_types = sorted(set(obj["objectType"] for obj in objects)) 
print(object_types)

controller_lock = threading.RLock()

nav_status = {
    "running": False,
    "message": "Ready",
    "step": 0,
    "target_place": None,
    "target_object": None,
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
    <title>RoboTHOR Object Navigation</title>
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
    <h2>RoboTHOR Object Navigation</h2>

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
            "step: " + data.step + "\\n" +
            "target_place: " + data.target_place + "\\n" +
            "target_object: " + data.target_object + "\\n" +
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


def find_existing_target_object(place_label):
    candidates = PLACE_TO_OBJECTS.get(place_label)

    if candidates is None:
        return None

    objects = get_last_event().metadata["objects"]

    for target_type in candidates:
        matched = [
            obj for obj in objects
            if obj["objectType"] == target_type
        ]

        if matched:
            return matched[0]

    return None

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

def nearest_reachable_position(object_pos, reachable):
    ox, oz = round_pos(object_pos)

    return min(
        reachable,
        key=lambda p: (p[0] - ox) ** 2 + (p[1] - oz) ** 2
    )


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
    nav_status["target_place"] = place_label
    nav_status["target_object"] = None
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

        target_object = find_existing_target_object(place_label)

        if target_object is None:
            msg = f"No target object found for place: {place_label}"
            nav_status["message"] = msg
            log(msg)
            nav_status["running"] = False
            return

        goal = nearest_reachable_position(
            target_object["position"],
            reachable
        )

        path = bfs(start, goal, reachable)

        nav_status["target_object"] = target_object["objectType"]
        nav_status["start"] = start
        nav_status["goal"] = goal
        nav_status["path_length"] = len(path)

        log(f"target object: {target_object['objectType']}")
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

        nav_status["message"] = (
            f"Going to {place_label} via {target_object['objectType']}"
        )

        follow_path(path)

        msg = f"Arrived near: {target_object['objectType']}"
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
            "\nPlace label ex) kitchen, bathroom, bedroom, living_room: "
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
