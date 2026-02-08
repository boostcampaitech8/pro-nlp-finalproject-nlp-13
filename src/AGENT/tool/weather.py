import os
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

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
        return {"error": f"'{city}' 도시를 찾을 수 없음."}
    except Exception as e:
        return {"error": f"geo API 오류: {e}."}
    
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
            "강수량": item.get("rain", {}).get("3h", 0),
            "적설량": item.get("snow", {}).get("3h", 0) / 10,
        })

    return {"coord": coord, "forecasts": forecasts}

def forecast_tool(city: str, days: int = 1) -> str:
    """
    도시명과 일수를 받아 예보 날씨 정보를 반환한다.

    Args:
        city: 날씨를 조회할 도시명 (예: "부산", "서울", "도쿄")
        days: 미래의 날씨 (예: 1은 내일, 2는 모레, 3은 글모레)
    """
    data = get_forecast(city)
    if "error" in data:
        return data["error"]

    coord = data["coord"]
    forecasts = data["forecasts"]
    
    if days > 365:
        target_date_obj = datetime.strptime(str(days), "%Y%m%d")
        target_date = target_date_obj.strftime("%Y-%m-%d")
    else:
        target_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    filtered = [f for f in forecasts if f["일시"].startswith(target_date)]

    if not filtered:
        return f"'{target_date}' 예보 데이터가 없습니다."
    
    perceived_temp = [f["체감"] for f in filtered]
    max_temp = max(f["섭씨"] for f in filtered)
    min_temp = min(f["섭씨"] for f in filtered)
    average_perceived_temp = sum(perceived_temp) / len(perceived_temp)
    total_rain = sum(f["강수량"] for f in filtered)
    total_snow = sum(f["적설량"] for f in filtered)
    
    prefix = f"{coord['error']}\n" if "error" in coord else ""

    result = (
        f"{prefix}{coord['name']} {target_date} 날씨 예보:\n\n"
        f"요약\n"
        f"- 최고기온: {max_temp}°C\n"
        f"- 최저기온: {min_temp}°C\n"
        f"- 체감 온도: {average_perceived_temp}°C\n"
        f"- 총 강수량: {total_rain}mm\n"
        f"- 총 적설량: {total_snow}cm\n\n"
    )
    return result