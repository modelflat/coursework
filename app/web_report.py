from collections import defaultdict
from multiprocessing import Lock

from flask import Flask, render_template, request, redirect
from flask_socketio import SocketIO, emit, send

app = Flask(__name__)
io = SocketIO(app)


class JobRegistry:

    def __init__(self):
        self.jobs = set()
        self.job_logs = defaultdict(lambda: [])
        self.lock = Lock()
        self.current_job_id = 0

    def _new_job_id(self):
        with self.lock:
            self.current_job_id += 1
            jid = "job_" + str(self.current_job_id)
            self.jobs.add(jid)
            return jid

    def run_job(self, params):
        job_id = self._new_job_id()

        def append_tqdm_line(line):
            with self.lock:
                self.job_logs[job_id].append(line)

        # ???

        return job_id

    def get_logs(self, job_id, last_id=0):
        logs = self.job_logs[job_id]
        return logs[last_id:], len(logs)


registry = JobRegistry()


@app.route("/<job_id>")
def job(job_id):
    return render_template("job.html", job_id=job_id)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        return redirect(registry.run_job(request.form["json"]))


@io.on("get_state", namespace="/")
def get_state():
    send("")
