import asyncio
import os

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from dotenv import load_dotenv

import google.generativeai as genai

# --- LOAD ENV ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- GEMINI CONFIG ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# --- TELEGRAM ---
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "🤖 Gemini бот онлайн.\n"
        "Пиши любой вопрос — отвечу."
    )


@dp.message(F.text)
async def chat(message: Message):
    try:
        response = model.generate_content(message.text)

        if response and response.text:
            await message.answer(response.text)
        else:
            await message.answer("🤖 Я задумался… попробуй иначе сформулировать")

    except Exception as e:
        await message.answer("💀 Gemini упал, но мы живы")
        print("ERROR:", e)


async def main():
    print("BOT STARTED")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
