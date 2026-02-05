"""LangGraph Tools - 날씨 조회 도구들"""
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool

from .weather_api import get_weather, get_forecast, get_history, parse_history

KST = timezone(timedelta(hours=9))


@tool
def weather_tool(city: str) -> str:
    """
    도시명을 받아 현재 날씨 정보를 반환한다.

    Args:
        city: 날씨를 조회할 도시명 (예: "부산", "서울", "도쿄")
    """
    weather = get_weather(city)

    if "error" in weather:
        return weather["error"]

    coord = weather.pop("coord")
    prefix = f"⚠️ {coord['error']}\n" if "error" in coord else ""

    return (
        f"{prefix}"
        f"{coord['name']} 현재 날씨:\n"
        f"- 온도: {weather['temp']}°C (체감 {weather['feels_like']}°C)\n"
        f"- 습도: {weather['humidity']}%\n"
        f"- 날씨: {weather['description']}\n"
        f"- 풍속: {weather['wind_speed']} m/s\n"
        f"- 강수량: {weather['rain']}mm\n"
        f"- 적설량: {weather['snow']}mm\n"
        f"- 구름양: {weather['clouds']}%"
    )


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

    target_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    filtered = [f for f in forecasts if f["일시"].startswith(target_date)]

    if not filtered:
        return f"'{target_date}' 예보 데이터가 없습니다."

    max_temp = max(f["섭씨"] for f in filtered)
    min_temp = min(f["섭씨"] for f in filtered)
    total_rain = sum(f["강수량"] for f in filtered)
    total_snow = sum(f["적설량"] for f in filtered)

    prefix = f"⚠️ {coord['error']}\n" if "error" in coord else ""

    result = (
        f"{prefix}{coord['name']} {target_date} 날씨 예보:\n\n"
        f"요약\n"
        f"- 최고기온: {max_temp}°C\n"
        f"- 최저기온: {min_temp}°C\n"
        f"- 총 강수량: {total_rain}mm\n"
        f"- 총 적설량: {total_snow}cm\n\n"
    )

    for f in filtered:
        result += (
            f"[{f['일시']}]\n"
            f"- 온도: {f['섭씨']}°C (체감 {f['체감']}°C)\n"
            f"- 습도: {f['습도']}%\n"
            f"- 날씨: {f['날씨']}\n"
            f"- 풍속: {f['풍속']} m/s\n"
            f"- 강수량: {f['강수량']}mm\n"
            f"- 적설량: {f['적설량']}cm\n"
            f"- 구름양: {f['구름양']}%\n\n"
        )
    return result


@tool
def history_tool(city: str, days_ago: int = 1) -> str:
    """
    과거 날씨 데이터를 조회하는 툴.

    Args:
        city: 도시명 (한글 가능, 예: "서울", "부산")
        days_ago: 몇 날 전의 날씨 (1=어제, 2=그제, ... 최대 7)
    """
    if days_ago < 1 or days_ago > 7:
        return "학생 플랜은 최대 1주일(7일) 이내의 과거 날씨만 조회 가능합니다."

    data = get_history(city, days_ago=days_ago)
    if not data:
        return f"{city}의 과거 날씨 데이터를 가져올 수 없습니다."

    parsed = parse_history(data)
    if not parsed:
        return "데이터 파싱에 실패했습니다."

    target_date = (datetime.now(KST) - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    result = (
        f"📅 {city} - {target_date} 날씨\n"
        f"🌡️ 최고기온: {parsed['max_temp']}°C / 최저기온: {parsed['min_temp']}°C\n"
        f"🌧️ 총강수량: {parsed['total_rain_mm']}mm\n"
        f"❄️ 총적설량: {parsed['total_snow_mm']}mm\n\n"
        f"시간별 상세:\n"
    )

    for h in parsed["hourly"]:
        result += f"  {h['time']} | {h['temp']}°C | {h['description']} | 풍속 {h['wind_speed']}m/s\n"

    return result
