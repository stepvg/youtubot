# YouTube Summarizer Telegram Bot

[Project link](https://github.com/stepvg/youtubot)

## Overview

This Telegram bot utilizes Large Language Models (LLM) for summarizing and answering questions about YouTube videos. It provides users with concise summaries and responses based on the content of the videos.

## Features

- Summarize YouTube videos: The bot can generate brief summaries of YouTube videos.
- Answer questions: Users can ask questions about the video content, and the bot will attempt to provide relevant answers.
- Telegram integration: The bot is integrated with Telegram messenger, allowing users to interact with it seamlessly.

## Deployment

The application is deployed using Docker Compose, with two containers for Redis and Python.

### Prerequisites

- Docker
- Docker Compose

### Deployment Steps

1. Install Git and Docker Compose
```bash
sudo apt install git docker-compose
```

2. Add current user to Docker group
```bash
sudo usermod $(whoami) -a -G docker
```

3. Clone the current repository
```bash
git clone https://github.com/stepvg/youtubot.git
```

4. Navigate to the project directory:
```bash
cd youtubot/
```

5. Add a Telegram token and change the necessary parameters in the .env file.
```bash
editor .env
```

6. Run containers
```bash
docker-compose up -d --build
```

7. The bot should now be running and accessible via Telegram.

## Usage

1. Start a conversation with the bot on Telegram.
2. Send a YouTube video link to the bot.
3. Optionally, ask questions about the video content.
4. The bot will respond with a summary or answers based on the provided video.

## Configuration

The bot's behavior and settings can be configured via environment variables or a `.env` file. The following variables are available:

- `HUGGING_FACE_EMBEDDINGS`: The name of the embeddings model, which can be obtained from the [huggingface](https://huggingface.co/).
- `TELEGRAM_BOT_TOKEN`: Telegram bot token obtained from BotFather.
- `GIGACHAT_CREDENTIALS`: API key for accessing the GigaChat Large Language Model service.
- `REDIS_HOST`: Hostname for the Redis container.
- `REDIS_PORT`: Port for accessing the Redis container.
- `REDIS_PASSWORD`: Password for accessing the Redis container.

## Contributing

Contributions to the project are welcome! Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Source code
* [Docker Compose file](docker-compose.yml) describes the Redis server and the bot application.
* [Python file](youtubot/main.py) contains the entry point to the bot application.
* [Dockerfile](youtubot/Dockerfile) describes the Docker image that is used to run the bot application.

