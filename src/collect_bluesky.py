import datetime
import pandas as pd
import requests

# 1. User Credentials
BSKY_HANDLE = "ves-perrr.bsky.social"  # Replace with your handle
BSKY_APP_PASSWORD = "7gux-bicc-tssl-ekxs"  # Replace with generated App Password

# 2. Endpoints & Config
AUTH_URL = "https://bsky.social/xrpc/com.atproto.server.createSession"
SEARCH_URL = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"

KEYWORDS = [
    "Samsung Galaxy S24",
    "Samsung Galaxy S23",
    "Samsung Z Fold",
    "Samsung Unpacked",
]
MAX_POSTS_PER_KEYWORD = 500

collected_data = []

print("Authenticating with Bluesky...")
headers = {"User-Agent": "SamsungSentimentResearch/1.0"}

# Step 1: Authenticate to get access JWT token
auth_resp = requests.post(
    AUTH_URL,
    json={"identifier": BSKY_HANDLE, "password": BSKY_APP_PASSWORD},
    headers=headers,
)

if auth_resp.status_code != 200:
    print(
        f"Authentication failed ({auth_resp.status_code}): {auth_resp.text}"
    )
    exit()

access_token = auth_resp.json().get("accessJwt")
headers["Authorization"] = f"Bearer {access_token}"
print("Authentication successful!\n")

# Step 2: Fetch Posts
for query in KEYWORDS:
    print(f"Fetching posts for: '{query}'...")
    params = {"q": query, "limit": 100}
    cursor = None
    query_count = 0

    while query_count < MAX_POSTS_PER_KEYWORD:
        if cursor:
            params["cursor"] = cursor

        try:
            res = requests.get(SEARCH_URL, headers=headers, params=params)
            if res.status_code != 200:
                print(f"  [API Error] Status Code {res.status_code}")
                break

            posts = res.json().get("posts", [])
            if not posts:
                break

            for p in posts:
                record = p.get("record", {})
                raw_date = record.get("createdAt")

                if raw_date and record.get("text"):
                    dt_obj = datetime.datetime.fromisoformat(
                        raw_date.replace("Z", "+00:00")
                    )

                    collected_data.append(
                        {
                            "id": p.get("cid"),
                            "platform": "bluesky",
                            "type": "post",
                            "created_at": dt_obj.strftime("%Y-%m-%d %H:%M:%S"),
                            "text": record.get("text").strip(),
                            "engagement_score": p.get("likeCount", 0),
                            "source_title": f"Search: {query}",
                            "parent_id": None,
                        }
                    )
                    query_count += 1

            cursor = res.json().get("cursor")
            if not cursor:
                break

        except Exception as e:
            print(f"  [Error] {e}")
            break

# Step 3: Save output
df_bsky = pd.DataFrame(collected_data)
if not df_bsky.empty:
    df_bsky = df_bsky.dropna(subset=["text"]).reset_index(drop=True)
    df_bsky.to_csv("bluesky_samsung_sentiment_data.csv", index=False)
    print(
        f"\nSuccess! Saved {len(df_bsky)} records to 'bluesky_samsung_sentiment_data.csv'."
    )
else:
    print("\nNo Bluesky records collected.")