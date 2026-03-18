"""
Telegram-бот рекомендаций событий
Стек: aiogram 3, JSON-база событий, OpenAI API
Два режима: кнопочный (теги → дата) и живой текстовый запрос (/ask)
"""

import json
import logging
import os
from datetime import datetime, timedelta
from openai import AsyncOpenAI

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv


# ─── Конфиг ────────────────────────────────────────────────────────────────────
load_dotenv()

BOT_TOKEN  = os.getenv("BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

EVENTS_FILE    = "events.json"
WELCOME_IMAGE  = "welcome.PNG"
RESEARCH_IMAGE = "research.PNG"
FIND_IMAGE     = "find.PNG"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── Загрузка событий ──────────────────────────────────────────────────────────

def load_events() -> list[dict]:
    with open(EVENTS_FILE, encoding="utf-8") as f:
        return json.load(f)

EVENTS = load_events()

# ─── Все доступные теги ────────────────────────────────────────────────────────

ALL_TAGS = {
    "education":     "📚 Образование",
    "tech":          "💻 Технологии",
    "business":      "💼 Бизнес",
    "startup":       "🚀 Стартапы",
    "networking":    "🤝 Нетворкинг",
    "career":        "🎯 Карьера",
    "art":           "🎨 Искусство",
    "music":         "🎵 Музыка",
    "science":       "🔬 Наука",
    "entertainment": "🎭 Развлечения",
    "sport":         "⚽ Спорт",
    "food":          "🍕 Еда",
    "free":          "🆓 Бесплатно",
}

# ─── FSM: состояния диалога ────────────────────────────────────────────────────

class Form(StatesGroup):
    choosing_tags = State()   # Пользователь выбирает теги
    choosing_date = State()   # Пользователь выбирает дату
    free_query    = State()   # Ожидаем свободный текстовый запрос

# ─── Хранилище выбранных тегов ─────────────────────────────────────────────────

user_tags: dict[int, set[str]] = {}

# ─── Фильтрация событий (для кнопочного режима) ────────────────────────────────

def filter_events(tags: set[str], date_from: str, date_to: str) -> list[dict]:
    result = []
    for event in EVENTS:
        if not (date_from <= event["date"] <= date_to):
            continue
        if tags and not tags.intersection(set(event["tags"])):
            continue
        result.append(event)
    result.sort(key=lambda e: e["date"])
    return result[:5]

# ─── Форматирование без ИИ (fallback) ─────────────────────────────────────────

def format_events_plain(events: list[dict]) -> str:
    if not events:
        return "По вашим критериям ничего не нашлось. Попробуйте изменить запрос."
    lines = ["Вот что нашлось для вас:\n"]
    for i, e in enumerate(events, 1):
        lines.append(
            f"{i}. {e['title']}\n"
            f"   📅 {e['date']} в {e['time']}\n"
            f"   📍 {e['location']}\n"
            f"   💸 {e.get('price', 'цена не указана')}\n"
            f"   🔗 {e['link']}\n"
        )
    return "\n".join(lines)

# ─── LLM: кнопочный режим (теги + дата → красивые карточки) ───────────────────

async def llm_recommend(events: list[dict], selected_tags: set[str]) -> str:
    if not OPENAI_KEY or not events:
        return format_events_plain(events)

    client = AsyncOpenAI(api_key=OPENAI_KEY)

    events_data = "\n".join([
        f"{i+1}. Название: {e['title']}\n"
        f"   Дата: {e['date']} в {e['time']}\n"
        f"   Место: {e['location']}\n"
        f"   Цена: {e.get('price', 'не указана')}\n"
        f"   Теги: {', '.join(e.get('tags', []))}"
        for i, e in enumerate(events)
    ])
    tags_text = ", ".join([ALL_TAGS.get(t, t) for t in selected_tags])

    prompt = f"""Ты — дружелюбный помощник по досугу в Telegram-боте.
Пользователь интересуется: {tags_text}.

Вот найденные события (их {len(events)}):
{events_data}

Верни ТОЛЬКО валидный JSON и ничего кроме него:
{{
  "intro": "1-2 живых предложения — как будто советуешь другу. Упомяни общую тему подборки.",
  "hooks": [
    "крючок для события 1: одно предложение почему стоит сходить",
    "крючок для события 2"
  ]
}}

Правила:
- intro — тёплое, личное, без перечисления названий
- hooks — ровно {len(events)} штук, в том же порядке что события
- только JSON, без ```json``` обёртки и любого текста снаружи"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.75,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ai = json.loads(raw.strip())

        intro = ai.get("intro", "")
        hooks = ai.get("hooks", [])

        lines = [f"✨ {intro}\n"]
        for i, e in enumerate(events):
            hook = hooks[i] if i < len(hooks) else ""
            lines.append(
                f"{'─' * 28}\n"
                f"<b>{i+1}. {e['title']}</b>\n"
                f"📅 {e['date']} в {e['time']}\n"
                f"📍 {e['location']}\n"
                f"💸 {e.get('price', 'цена не указана')}\n"
                + (f"💬 <i>{hook}</i>\n" if hook else "")
                + f"🔗 {e['link']}"
            )
        return "\n".join(lines)

    except Exception as e:
        log.warning(f"LLM error: {e}")
        return format_events_plain(events)


# ─── LLM: СВОБОДНЫЙ ТЕКСТОВЫЙ ЗАПРОС ──────────────────────────────────────────

async def llm_free_search(user_query: str) -> str:
    """
    Пользователь пишет что угодно на живом языке.
    ИИ сам понимает запрос, выбирает подходящие события из базы и отвечает.
    """
    if not OPENAI_KEY:
        return "⚠️ Режим живого поиска недоступен: не настроен OpenAI API ключ."

    client = AsyncOpenAI(api_key=OPENAI_KEY)
    today = datetime.today().strftime("%Y-%m-%d")

    # Передаём ВСЕ события в ИИ — пусть сам выбирает подходящие
    all_events_data = json.dumps(EVENTS, ensure_ascii=False, indent=2)

    system_prompt = f"""Ты — умный помощник по досугу в Telegram-боте. Сегодняшняя дата: {today}.
У тебя есть база событий в Москве в формате JSON.

Твоя задача:
1. Понять запрос пользователя (тема, настроение, бюджет, дата, формат — что угодно)
2. Выбрать из базы до 5 наиболее подходящих событий
3. Вернуть ТОЛЬКО валидный JSON без обёрток

Формат ответа:
{{
  "understood": "одно предложение — как ты понял запрос пользователя",
  "events": [список индексов подходящих событий из массива, 0-based, максимум 5],
  "intro": "1-2 живых предложения — персональный совет другу под этот конкретный запрос",
  "hooks": ["почему событие подходит под запрос (1 предложение)", ...]
}}

Если ничего не подошло — верни: {{"understood": "...", "events": [], "intro": "", "hooks": []}}

Учитывай:
- слова про бюджет («бесплатно», «дёшево», «до 500 руб») → смотри поле price
- слова про время («сегодня», «в эти выходные», «на этой неделе») → считай от {today}
- настроение («потусить», «чему-то научиться», «познакомиться») → подбирай по смыслу
- можно комбинировать несколько критериев одновременно

База событий:
{all_events_data}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            max_tokens=900,
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ai = json.loads(raw.strip())

        event_indices = ai.get("events", [])
        intro         = ai.get("intro", "")
        hooks         = ai.get("hooks", [])
        understood    = ai.get("understood", "")

        if not event_indices:
            return (
                f"🤔 <i>{understood}</i>\n\n"
                "К сожалению, ничего подходящего не нашлось.\n\n"
                "Попробуй переформулировать — например:\n"
                "• <i>«что-нибудь бесплатное на выходных»</i>\n"
                "• <i>«хочу на концерт или выставку»</i>\n"
                "• <i>«митап по технологиям на этой неделе»</i>"
            )

        found = [EVENTS[i] for i in event_indices if 0 <= i < len(EVENTS)]

        lines = [
            f"🎯 <i>Понял тебя: {understood}</i>\n",
            f"✨ {intro}\n"
        ]
        for i, e in enumerate(found):
            hook = hooks[i] if i < len(hooks) else ""
            lines.append(
                f"{'─' * 28}\n"
                f"<b>{i+1}. {e['title']}</b>\n"
                f"📅 {e['date']} в {e['time']}\n"
                f"📍 {e['location']}\n"
                f"💸 {e.get('price', 'цена не указана')}\n"
                + (f"💬 <i>{hook}</i>\n" if hook else "")
                + f"🔗 {e['link']}"
            )
        return "\n".join(lines)

    except Exception as e:
        log.warning(f"LLM free search error: {e}")
        return "😔 Что-то пошло не так при обработке запроса. Попробуй ещё раз или используй /start."


# ─── Клавиатуры ────────────────────────────────────────────────────────────────

def tags_keyboard(selected: set[str]) -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    for tag, label in ALL_TAGS.items():
        mark = "✅ " if tag in selected else ""
        builder.button(text=f"{mark}{label}", callback_data=f"tag:{tag}")
    builder.button(text="➡️ Готово", callback_data="tags_done")
    builder.adjust(2)
    return builder

def date_keyboard() -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    builder.button(text="Сегодня",          callback_data="date:today")
    builder.button(text="Завтра",           callback_data="date:tomorrow")
    builder.button(text="Эта неделя",       callback_data="date:week")
    builder.button(text="Следующие 2 нед",  callback_data="date:2weeks")
    builder.button(text="Весь апрель",      callback_data="date:april")
    builder.button(text="Любая дата",       callback_data="date:any")
    builder.adjust(2)
    return builder

def after_results_keyboard() -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    builder.button(text="🔄 Изменить теги",      callback_data="restart_tags")
    builder.button(text="📅 Другой период",      callback_data="restart_date")
    builder.button(text="🗣 Спросить свободно",  callback_data="go_ask")
    builder.adjust(2)
    return builder


# ─── Bot & Dispatcher ──────────────────────────────────────────────────────────

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


# ─── /start ────────────────────────────────────────────────────────────────────

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    user_tags[message.from_user.id] = set()

    try:
        photo = FSInputFile(WELCOME_IMAGE)
        await message.answer_photo(photo)
    except Exception:
        pass

    builder = InlineKeyboardBuilder()
    builder.button(text="🎯 Выбрать по темам",  callback_data="mode_tags")
    builder.button(text="🗣 Написать свободно",  callback_data="mode_ask")
    builder.adjust(1)

    await message.answer(
        "Я помогу тебе найти интересные события в Москве ٩(◕‿◕)۶\n\n"
        "Как будем искать?",
        reply_markup=builder.as_markup()
    )


# ─── Выбор режима ──────────────────────────────────────────────────────────────

@dp.callback_query(F.data == "mode_tags")
async def mode_tags(callback: CallbackQuery, state: FSMContext):
    uid = callback.from_user.id
    user_tags[uid] = set()
    await callback.message.edit_text(
        "Выбери темы, которые тебя интересуют:",
        reply_markup=tags_keyboard(set()).as_markup()
    )
    await state.set_state(Form.choosing_tags)
    await callback.answer()


@dp.callback_query(F.data == "mode_ask")
async def mode_ask_callback(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Form.free_query)
    await callback.message.edit_text(
        "🗣 <b>Режим живого поиска</b>\n\n"
        "Напиши что ищешь своими словами. Например:\n\n"
        "• <i>«хочу что-нибудь бесплатное на выходных»</i>\n"
        "• <i>«ищу тусовку для стартаперов или нетворкинг»</i>\n"
        "• <i>«что-то интересное по технологиям на этой неделе»</i>\n"
        "• <i>«концерт или выставка, желательно до 1000 руб»</i>\n\n"
        "Пиши — я разберусь! 👇",
        parse_mode="HTML"
    )
    await callback.answer()


@dp.callback_query(F.data == "go_ask")
async def go_ask_callback(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Form.free_query)
    await callback.message.answer(
        "🗣 Напиши что ищешь своими словами — я подберу события:"
    )
    await callback.answer()


# ─── /ask ──────────────────────────────────────────────────────────────────────

@dp.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext):
    await state.set_state(Form.free_query)
    await message.answer(
        "🗣 <b>Режим живого поиска</b>\n\n"
        "Напиши что ищешь своими словами. Например:\n\n"
        "• <i>«хочу что-нибудь бесплатное на выходных»</i>\n"
        "• <i>«ищу тусовку для стартаперов или нетворкинг»</i>\n"
        "• <i>«что-то интересное по технологиям на этой неделе»</i>\n\n"
        "Пиши — я разберусь! 👇",
        parse_mode="HTML"
    )


# ─── Обработка свободного запроса ─────────────────────────────────────────────

@dp.message(Form.free_query)
async def handle_free_query(message: Message, state: FSMContext):
    user_query = message.text.strip()
    thinking = await message.answer("🤔 Анализирую твой запрос...")
    result = await llm_free_search(user_query)
    await thinking.delete()
    await message.answer(result, parse_mode="HTML")

    builder = InlineKeyboardBuilder()
    builder.button(text="🗣 Спросить ещё",       callback_data="go_ask")
    builder.button(text="🎯 Выбрать по темам",   callback_data="mode_tags")
    builder.button(text="🏠 В начало",           callback_data="go_home")
    builder.adjust(2)

    await message.answer("Хочешь найти что-то ещё?", reply_markup=builder.as_markup())
    await state.clear()


# ─── /help ─────────────────────────────────────────────────────────────────────

@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "ℹ️ <b>Как пользоваться ботом:</b>\n\n"
        "/start — начать заново\n"
        "/ask   — живой поиск: пишешь что хочешь, ИИ подбирает\n"
        "/find  — найти по тегам (если интересы уже выбраны)\n"
        "/tags  — посмотреть выбранные теги\n\n"
        "<b>Два режима поиска:</b>\n"
        "🎯 <i>По темам</i> — выбираешь теги → период → получаешь подборку\n"
        "🗣 <i>Свободный запрос</i> — пишешь на живом языке, ИИ сам разбирается",
        parse_mode="HTML"
    )


# ─── /tags ─────────────────────────────────────────────────────────────────────

@dp.message(Command("tags"))
async def cmd_tags(message: Message):
    selected = user_tags.get(message.from_user.id, set())
    if not selected:
        await message.answer("Ты ещё не выбирал теги. Нажми /start чтобы начать.")
        return
    labels = [ALL_TAGS[t] for t in selected if t in ALL_TAGS]
    await message.answer(f"Твои интересы: {', '.join(labels)}")


# ─── /find ─────────────────────────────────────────────────────────────────────

@dp.message(Command("find"))
async def cmd_find(message: Message, state: FSMContext):
    selected = user_tags.get(message.from_user.id, set())
    if not selected:
        await message.answer("Сначала выбери интересы через /start")
        return
    await message.answer(
        "На какой период ищем события?",
        reply_markup=date_keyboard().as_markup()
    )
    await state.set_state(Form.choosing_date)


# ─── Выбор тегов ──────────────────────────────────────────────────────────────

@dp.callback_query(Form.choosing_tags, F.data.startswith("tag:"))
async def toggle_tag(callback: CallbackQuery, state: FSMContext):
    tag = callback.data.split(":")[1]
    uid = callback.from_user.id
    if uid not in user_tags:
        user_tags[uid] = set()
    if tag in user_tags[uid]:
        user_tags[uid].discard(tag)
    else:
        user_tags[uid].add(tag)
    selected = user_tags[uid]
    count = len(selected)
    hint = f"Выбрано: {count}" if count > 0 else "Выбери хотя бы один интерес"
    await callback.message.edit_text(
        f"Выбери темы, которые тебя интересуют:\n\n{hint}",
        reply_markup=tags_keyboard(selected).as_markup()
    )
    await callback.answer()


@dp.callback_query(Form.choosing_tags, F.data == "tags_done")
async def tags_done(callback: CallbackQuery, state: FSMContext):
    selected = user_tags.get(callback.from_user.id, set())
    if not selected:
        await callback.answer("Выбери хотя бы один тег!", show_alert=True)
        return
    labels = [ALL_TAGS[t] for t in selected]
    await callback.message.edit_text(
        f"Запомнил твои интересы ᕦ(ಠ_ಠ)ᕤ\n{', '.join(labels)}\n\n"
        f"На какой период ищем события?",
        reply_markup=date_keyboard().as_markup()
    )
    await state.set_state(Form.choosing_date)
    await callback.answer()


# ─── Выбор даты и показ результата ───────────────────────────────────────────

@dp.callback_query(Form.choosing_date, F.data.startswith("date:"))
async def choose_date(callback: CallbackQuery, state: FSMContext):
    period = callback.data.split(":")[1]
    today  = datetime.today()

    if period == "today":
        date_from = date_to = today.strftime("%Y-%m-%d")
    elif period == "tomorrow":
        d = today + timedelta(days=1)
        date_from = date_to = d.strftime("%Y-%m-%d")
    elif period == "week":
        date_from = today.strftime("%Y-%m-%d")
        date_to   = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    elif period == "2weeks":
        date_from = today.strftime("%Y-%m-%d")
        date_to   = (today + timedelta(days=14)).strftime("%Y-%m-%d")
    elif period == "april":
        date_from = "2026-04-01"
        date_to   = "2026-04-30"
    else:
        date_from = "2026-01-01"
        date_to   = "2026-12-31"

    uid      = callback.from_user.id
    selected = user_tags.get(uid, set())

    await callback.message.edit_text("🔍 Ищу события...")
    found = filter_events(selected, date_from, date_to)

    if not found:
        await callback.message.edit_text(
            "По твоим критериям ничего не нашлось (ಥ﹏ಥ)\n\n"
            "Попробуй:\n"
            "• выбрать другой период\n"
            "• добавить больше тегов\n"
            "• или напиши запрос свободно через /ask\n\n"
            "Нажми /start чтобы начать заново."
        )
        await state.clear()
        await callback.answer()
        return

    await callback.message.edit_text("✨ Формирую рекомендации...")
    text = await llm_recommend(found, selected)
    await callback.message.edit_text(text, parse_mode="HTML")
    await callback.message.answer(
        "Хочешь найти что-то ещё? (° ͜ʖ ͡°)",
        reply_markup=after_results_keyboard().as_markup()
    )
    await state.clear()
    await callback.answer()


# ─── Рестарт ──────────────────────────────────────────────────────────────────

@dp.callback_query(F.data == "restart_tags")
async def restart_tags(callback: CallbackQuery, state: FSMContext):
    uid = callback.from_user.id
    user_tags[uid] = set()
    await callback.message.edit_text(
        "Выбери темы, которые тебя интересуют:",
        reply_markup=tags_keyboard(set()).as_markup()
    )
    await state.set_state(Form.choosing_tags)
    await callback.answer()


@dp.callback_query(F.data == "restart_date")
async def restart_date(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text(
        "На какой период ищем события?",
        reply_markup=date_keyboard().as_markup()
    )
    await state.set_state(Form.choosing_date)
    await callback.answer()


@dp.callback_query(F.data == "go_home")
async def go_home(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    user_tags[callback.from_user.id] = set()
    builder = InlineKeyboardBuilder()
    builder.button(text="🎯 Выбрать по темам",  callback_data="mode_tags")
    builder.button(text="🗣 Написать свободно",  callback_data="mode_ask")
    builder.adjust(1)
    await callback.message.answer(
        "Как будем искать?",
        reply_markup=builder.as_markup()
    )
    await callback.answer()


# ─── Запуск ───────────────────────────────────────────────────────────────────

async def main():
    log.info("Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
