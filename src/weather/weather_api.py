"""OpenWeather API 호출 함수들"""
import requests
from datetime import datetime, timedelta, timezone

from .weather_config import DEFAULT_CITY
from dotenv import load_dotenv
import os

KST = timezone(timedelta(hours=9))
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_coordinates(city: str) -> dict:
    """도시명으로 좌표 조회"""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={WEATHER_API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            name = data.get("local_names", {}).get("ko", data["name"])
            return {"name": name, "lat": data["lat"], "lon": data["lon"]}
        return {"error": f"'{city}' 도시를 찾을 수 없음.", **DEFAULT_CITY}
    except Exception as e:
        return {"error": f"geo API 오류: {e}.", **DEFAULT_CITY}


def get_weather(city: str) -> dict:
    """현재 날씨 조회"""
    coord = get_coordinates(city)

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={coord['lat']}&lon={coord['lon']}"
        f"&appid={WEATHER_API_KEY}&units=metric&lang=kr"
    )
    response = requests.get(url, timeout=5)

    if response.status_code != 200:
        return {"error": f"Weather API 에러: {response.status_code}"}

    data = response.json()
    return {
        "coord": coord,
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
        "rain": data.get("rain", {}).get("1h", 0),
        "snow": data.get("snow", {}).get("1h", 0),
        "clouds": data["clouds"]["all"],
    }


def get_forecast(city: str) -> dict:
    """5일간 3시간 간격 예보 조회"""
    coord = get_coordinates(city)

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={coord['lat']}&lon={coord['lon']}"
        f"&appid={WEATHER_API_KEY}&units=metric&lang=kr"
    )
    response = requests.get(url, timeout=5)

    if response.status_code != 200:
        return {"error": f"Forecast API 에러: {response.status_code}"}

    data = response.json()
    forecasts = []
    for item in data["list"]:
        forecasts.append({
            "일시": item["dt_txt"],
            "섭씨": item["main"]["temp"],
            "체감": item["main"]["feels_like"],
            "습도": item["main"]["humidity"],
            "날씨": item["weather"][0]["description"],
            "풍속": item["wind"]["speed"],
            "강수량": item.get("rain", {}).get("3h", 0),
            "적설량": item.get("snow", {}).get("3h", 0) / 10,
            "구름양": item["clouds"]["all"],
        })

    return {"coord": coord, "forecasts": forecasts}


def get_history(city: str, days_ago: int = 1, hours: int = 24) -> dict | None:
    """과거 날씨 데이터 조회 (학생 플랜: 최대 1주일)"""
    coord = get_coordinates(city)
    if not coord:
        return None

    # start 타임스탬프 계산: 해당 날짜 00:00 KST
    now_kst = datetime.now(KST)
    target_start = (
        now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
        - timedelta(days=days_ago)
    )
    start_unix = int(target_start.timestamp())

    url = (
        f"https://history.openweathermap.org/data/2.5/history/city"
        f"?lat={coord['lat']}&lon={coord['lon']}"
        f"&type=hour"
        f"&start={start_unix}"
        f"&cnt={hours}"
        f"&appid={WEATHER_API_KEY}"
        f"&units=metric"
        f"&lang=kr"
    )

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"History API error: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def parse_history(data: dict) -> dict:
    """History API 응답을 정리된 형태로 변환"""
    if not data or "list" not in data:
        return {}

    hourly = []
    for item in data["list"]:
        dt_kst = datetime.fromtimestamp(item["dt"], tz=KST)

        hourly.append({
            "time": dt_kst.strftime("%H:%M"),
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "description": item["weather"][0]["description"],
            "wind_speed": item["wind"]["speed"],
            "rain_1h": item.get("rain", {}).get("1h", 0.0),
            "snow_1h": item.get("snow", {}).get("1h", 0.0),
            "clouds": item["clouds"]["all"],
        })

    temps = [h["temp"] for h in hourly]

    return {
        "max_temp": max(temps) if temps else None,
        "min_temp": min(temps) if temps else None,
        "total_rain_mm": round(sum(h["rain_1h"] for h in hourly), 1),
        "total_snow_mm": round(sum(h["snow_1h"] for h in hourly), 1),
        "hourly": hourly,
    }
