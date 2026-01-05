import requests
import random
import time

# ================= CONFIG =================
ACCESS_TOKEN = "EAAMVrwZBedl8BQdhuxvFI7bQfnT1qqpdhWTuYhZC2lEjoMQ9y41YQI42R7UhAed8qVazDmXGQ7LUU31XUCNMOQdD6WGjrC1CZBS3poIM4IRJAhTPEWQvdQMYt4mY54P2xxXhlmHL2cIZCLyVsQ6ZCQHQjpEyLH4hwL6KwGq5khwL939rHL6Imr6o3ZA0XUZClggkJIZC2QWcCzgzuivTZBq0t7vbJVdTa8Ki1TCXq9wZDZD"
PAGE_NAME = "Gridiron Central"   # EXACT page name
GRAPH = "https://graph.facebook.com/v24.0"

EMOJIS = ["🔥", "❤️", "😂", "😍", "🙌", "💯", "👏", "😎", "✨"]

SLEEP_BETWEEN_REPLIES = 1.5
COMMENTS_LIMIT = 100
REPLIES_LIMIT = 50
# =========================================

stats = {
    "checked": 0,
    "replied": 0,
    "skipped": 0,
    "errors": 0
}


def get_comments(object_id):
    """Fetch comments with paging support"""
    endpoint = f"{GRAPH}/{object_id}/comments"
    params = {
        "access_token": ACCESS_TOKEN,
        "fields": f"id,message,from{{id,name}},comments.limit({REPLIES_LIMIT}){{from{{id,name}}}}",
        "limit": COMMENTS_LIMIT
    }

    all_comments = []

    while endpoint:
        r = requests.get(endpoint, params=params, timeout=15)
        res = r.json()

        if "error" in res:
            print("❌ API ERROR:", res["error"])
            break

        data = res.get("data", [])
        all_comments.extend(data)

        # Paging
        paging = res.get("paging", {})
        endpoint = paging.get("next")
        params = None  # next already has params baked in

    return all_comments


def already_replied(comment):
    """Check if page already replied"""
    replies = comment.get("comments", {}).get("data", [])
    for r in replies:
        if r.get("from", {}).get("name") == PAGE_NAME:
            return True
    return False


def reply_to_comment(comment_id):
    url = f"{GRAPH}/{comment_id}/comments"
    payload = {
        "access_token": ACCESS_TOKEN,
        "message": random.choice(EMOJIS)
    }

    r = requests.post(url, data=payload, timeout=10)

    if r.status_code not in (200, 201):
        try:
            print("❌ Reply error:", r.json())
        except Exception:
            print("❌ Reply failed:", r.status_code, r.text)
        return False

    return True


def process_object(object_id):
    print("\n==============================")
    print("🚀 SCANNING OBJECT")
    print(f"🔗 https://www.facebook.com/{object_id}")
    print("==============================")

    comments = get_comments(object_id)
    print(f"🧠 Comments found: {len(comments)}\n")

    for c in comments:
        stats["checked"] += 1
        cid = c["id"]
        link = f"https://www.facebook.com/{cid}"

        if already_replied(c):
            stats["skipped"] += 1
            print(f"⏭️  Skipped → {link}")
            continue

        ok = reply_to_comment(cid)

        if ok:
            stats["replied"] += 1
            print(f"✅ Replied → {link}")
        else:
            stats["errors"] += 1
            print(f"❌ Failed → {link}")

        time.sleep(SLEEP_BETWEEN_REPLIES)

    print("\n📊 FINAL STATS")
    print(f"Checked : {stats['checked']}")
    print(f"Replied : {stats['replied']}")
    print(f"Skipped : {stats['skipped']}")
    print(f"Errors  : {stats['errors']}")
    print("------------------------------")


def main():
    print("🤖 FB COMMENT BOT — ENHANCED MODE")

    while True:
        object_id = input("\nPaste POST / REEL ID (or 'exit'): ").strip()
        if object_id.lower() == "exit":
            print("Shutting down. Humanity survives another day.")
            break

        if not object_id:
            print("❌ Empty ID. Try again.")
            continue

        process_object(object_id)


if __name__ == "__main__":
    main()
