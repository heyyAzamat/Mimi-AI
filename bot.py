import asyncio
import os
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- LOAD ENV ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# --- Телеграм ---
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --- Загружаем модель (мини-версия для CPU/GPU) ---
# Например, используем MPT-7B-StoryLite (или любую маленькую модель)
MODEL_NAME = "mosaicml/mpt-7b-story-lite"

print("🚀 Загружаем модель, может занять пару минут...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
print("✅ Модель загружена")

# --- Telegram хэндлеры ---
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "🤖 Привет! Я твой локальный ИИ.\n"
        "Пиши что угодно, отвечу."
    )

@dp.message(F.text)
async def chat(message: Message):
    user_text = message.text

    # Генерация ответа
    inputs = tokenizer(user_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    await message.answer(reply)

# --- Запуск бота ---
async def main():
    print("🤖 Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
