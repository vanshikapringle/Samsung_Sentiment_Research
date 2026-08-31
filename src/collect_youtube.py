import datetime
import pandas as pd
from googleapiclient.discovery import build

# 1. Configuration
API_KEY = "AIzaSyCS_G-bNYTXbWgZTmKjVJBF9mrqC7di1Rc"

# Added the missing variable definition line below:
SEARCH_KEYWORDS = [
    "Samsung Galaxy S24 Ultra review",
    "Samsung Galaxy Z Fold 6 review",
    "Samsung Galaxy S23 Ultra long term review",
]

MAX_VIDEOS_PER_KEYWORD = 3
MAX_COMMENTS_PER_VIDEO = 100

youtube = build("youtube", "v3", developerKey=API_KEY)
collected_data = []

print("Starting YouTube sentiment data collection...\n")

# 2. Iterate through search queries
for query in SEARCH_KEYWORDS:
    print(f"Searching videos for: '{query}'...")
    search_response = (
        youtube.search()
        .list(
            q=query,
            part="id,snippet",
            type="video",
            maxResults=MAX_VIDEOS_PER_KEYWORD,
        )
        .execute()
    )

    for item in search_response.get("items", []):
        video_id = item["id"]["videoId"]
        video_title = item["snippet"]["title"]
        print(f" -> Fetching comments for: {video_title} (ID: {video_id})")

        # 3. Fetch top-level comment threads per video
        try:
            comment_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=MAX_COMMENTS_PER_VIDEO,
            )

            while comment_request:
                comment_response = comment_request.execute()

                for c_item in comment_response.get("items", []):
                    snippet = c_item["snippet"]["topLevelComment"]["snippet"]

                    # Standardize timestamp to UTC YYYY-MM-DD HH:MM:SS
                    raw_date = snippet["publishedAt"]
                    dt_obj = datetime.datetime.fromisoformat(
                        raw_date.replace("Z", "+00:00")
                    )

                    collected_data.append(
                        {
                            "id": c_item["id"],
                            "platform": "youtube",
                            "type": "comment",
                            "created_at": dt_obj.strftime("%Y-%m-%d %H:%M:%S"),
                            "text": snippet["textOriginal"].strip(),
                            "engagement_score": snippet["likeCount"],
                            "source_title": video_title,
                            "parent_id": video_id,
                        }
                    )

                # Fetch next page of comments if available
                comment_request = youtube.commentThreads().list_next(
                    comment_request, comment_response
                )

        except Exception as e:
            print(f"    [Skipped] Could not fetch comments: {e}")

# 4. Save output to pandas DataFrame & export CSV
df = pd.DataFrame(collected_data)

# Remove empty comments or null values
if not df.empty:
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df[df["text"] != ""].reset_index(drop=True)
    df.to_csv("youtube_samsung_sentiment_data.csv", index=False)
    print(
        f"\nSuccess! Saved {len(df)} records to 'youtube_samsung_sentiment_data.csv'."
    )
else:
    print("\nNo comments were collected. Please check your API key and quota.")