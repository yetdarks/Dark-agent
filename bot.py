import asyncio
import os
from datetime import datetime
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from google import genai
from google.genai import types as genai_types

# Токены (замените на свои или положите в переменные окружения)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ВАШ_ТЕЛЕГРАМ_ТОКЕН")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ВАШ_GEMINI_KEY")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 1. ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ (TOOLS)
# ==========================================

def get_current_time() -> str:
    """Возвращает точное текущее время и дату."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_math_calculation(expression: str) -> str:
    """Вычисляет математическое выражение. На вход принимает строку, например: '2**16' или '15 * 43'."""
    try:
        # Для безопасности в проде лучше использовать специализированные библиотеки (sympy/math)
        allowed_chars = set("0123456789+-*/()., ^")
        if not all(c in allowed_chars for c in expression.replace(" ", "")):
            return "Ошибка: выражение содержит недопустимые символы."
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"

# Список доступных функций для Gemini
AGENT_TOOLS = [get_current_time, run_math_calculation]

# Словарь для быстрого вызова по имени
TOOL_MAP = {
    "get_current_time": get_current_time,
    "run_math_calculation": run_math_calculation,
}

# ==========================================
# 2. АГЕНТСКИЙ ЦИКЛ (FUNCTION CALLING)
# ==========================================

async def run_agent_loop(user_prompt: str) -> str:
    """Запускает диалог с Gemini и обрабатывает вызовы инструментов."""
    
    # Инициализация сообщений (включая системный промпт)
    messages = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(
                text=f"Ты — автономный ИИ-помощник. Используй доступные инструменты, когда это необходимо.\nЗапрос: {user_prompt}"
            )]
        )
    ]
    
    config = genai_types.GenerateContentConfig(
        tools=AGENT_TOOLS,
        temperature=0.2,
    )

    # Цикл выполнения тулов (макс. 5 шагов рассуждений)
    for _ in range(5):
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=messages,
            config=config,
        )

        candidate = response.candidates[0]
        model_content = candidate.content
        messages.append(model_content)

        # Проверяем, вызвала ли модель какую-либо функцию
        tool_calls = [part.function_call for part in model_content.parts if part.function_call]
        
        # Если тулы не нужны, модель вернула финальный ответ
        if not tool_calls:
            return response.text

        # Если тулы вызваны — выполняем их
        tool_responses_parts = []
        for call in tool_calls:
            tool_name = call.name
            tool_args = dict(call.args) if call.args else {}
            
            # Вызов соответствующей Python-функции
            if tool_name in TOOL_MAP:
                func_result = TOOL_MAP[tool_name](**tool_args)
            else:
                func_result = f"Ошибка: инструмент {tool_name} не найден."

            tool_responses_parts.append(
                genai_types.Part.from_function_response(
                    name=tool_name,
                    response={"result": func_result}
                )
            )

        # Возвращаем результаты функций обратно модели
        messages.append(
            genai_types.Content(role="tool", parts=tool_responses_parts)
        )

    return "Превышен лимит шагов рассуждений агента."

# ==========================================
# 3. ОБРАБОТЧИКИ TELEGRAM
# ==========================================

@dp.message(CommandStart())
async def handle_start(message: types.Message):
    await message.answer("Агент готов к работе. Напишите задачу!")

@dp.message()
async def handle_text(message: types.Message):
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        reply = await run_agent_loop(message.text)
        await message.answer(reply)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")

# ==========================================
# 4. ЗАПУСК
# ==========================================

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
