import requests

url = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"

params = {
    "q": "samsung",
    "limit": 10
}

response = requests.get(url, params=params)

print("Status:", response.status_code)
print("URL:", response.url)
print("Response:", response.text[:1000])