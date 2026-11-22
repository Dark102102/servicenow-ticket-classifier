"""Minimal ServiceNow REST client: creates incidents pre-routed to the predicted assignment group.

Credentials are read from the project's .env file (git-ignored):
    SN_INSTANCE=https://devXXXXX.service-now.com
    SN_USER=admin
    SN_PASSWORD=...
"""
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests

from src.data import LABELS, ROOT

TRIAGE_GROUP = "Service Desk Triage"   # receives tickets the model is not confident about
GROUP_NAMES = LABELS + [TRIAGE_GROUP]


def load_env(path: Path = ROOT / ".env") -> dict:
    """Read KEY=VALUE lines from .env (environment variables take precedence)."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    env = {k: os.environ.get(k, values.get(k, "")) for k in ("SN_INSTANCE", "SN_USER", "SN_PASSWORD")}
    # Keep only https://host – a pasted login/browser URL (with paths or query parameters) must not be used as the API base.
    url = urlsplit(env["SN_INSTANCE"] if "://" in env["SN_INSTANCE"] else "https://" + env["SN_INSTANCE"])
    env["SN_INSTANCE"] = f"https://{url.netloc}" if url.netloc else ""
    return env


def is_configured() -> bool:
    env = load_env()
    return bool(env["SN_PASSWORD"]) and "devXXXXX" not in env["SN_INSTANCE"]


class ServiceNowClient:
    def __init__(self, instance: str, user: str, password: str, session: requests.Session | None = None):
        self.instance = instance.rstrip("/")
        self.session = session or requests.Session()
        self.session.auth = (user, password)
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
        self._group_ids: dict[str, str] = {}

    @classmethod
    def from_env(cls) -> "ServiceNowClient":
        env = load_env()
        if not is_configured():
            raise RuntimeError("ServiceNow is not configured: fill in SN_INSTANCE and SN_PASSWORD in .env")
        return cls(env["SN_INSTANCE"], env["SN_USER"], env["SN_PASSWORD"])

    def _request(self, method: str, table: str, **kwargs) -> dict:
        r = self.session.request(method, f"{self.instance}/api/now/table/{table}", timeout=30, **kwargs)
        if r.status_code == 401:
            raise RuntimeError("ServiceNow rejected the login (401): check SN_USER / SN_PASSWORD in .env")
        r.raise_for_status()
        if "json" not in r.headers.get("Content-Type", ""):
            raise RuntimeError("ServiceNow returned a web page instead of API data: the instance may be hibernating "
                               "(log in through the browser to wake it) or SN_INSTANCE is not the instance address")
        return r.json()["result"]

    def ping(self) -> str:
        """Return the logged-in user's name; raises if the instance is asleep, unreachable or credentials are wrong."""
        user = self._request("GET", "sys_user", params={"user_name": self.session.auth[0], "sysparm_limit": 1,
                                                         "sysparm_fields": "user_name,name"})
        return user[0]["name"] if user else self.session.auth[0]

    def ensure_groups(self, names=GROUP_NAMES) -> dict[str, str]:
        """Make sure an assignment group exists for every support team; return {name: sys_id}."""
        for name in names:
            found = self._request("GET", "sys_user_group", params={"name": name, "sysparm_fields": "sys_id,name",
                                                                    "sysparm_limit": 1})
            if found:
                self._group_ids[name] = found[0]["sys_id"]
            else:
                created = self._request("POST", "sys_user_group", json={
                    "name": name, "description": "Support team used by the AI ticket-routing demo"})
                self._group_ids[name] = created["sys_id"]
        return dict(self._group_ids)

    def create_incident(self, text: str, predicted: str, confidence: float, threshold: float, model_name: str) -> dict:
        """Create an incident assigned to the predicted group (or to triage when confidence < threshold)."""
        auto = confidence >= threshold
        group = predicted if auto else TRIAGE_GROUP
        if group not in self._group_ids:
            self.ensure_groups([group])
        note = (f"[AI routing] Model: {model_name}\n"
                f"Predicted support group: {predicted} (confidence {confidence:.0%}, threshold {threshold:.0%})\n"
                + (f"Decision: auto-routed to {predicted}." if auto else
                   f"Decision: confidence below threshold - sent to {TRIAGE_GROUP} for human review "
                   f"(model's best guess: {predicted})."))
        short = " ".join(text.split())
        result = self._request("POST", "incident", json={
            "short_description": short[:157] + "..." if len(short) > 160 else short,
            "description": text,
            "assignment_group": self._group_ids[group],
            "work_notes": note,
        })
        return {"number": result["number"], "sys_id": result["sys_id"], "group": group, "auto": auto,
                "url": f"{self.instance}/incident.do?sys_id={result['sys_id']}"}
