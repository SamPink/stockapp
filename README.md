# Stock Sentiment Analysis App

The Stock Sentiment Analysis App is an innovative tool designed to gauge the market's mood on any given stock by scrutinizing the latest news and stock data. It utilizes the cognitive processing power of OpenAI's GPT-4 to deliver insights into stock sentiment, making it an intriguing asset for traders and investors alike.

## Concept
At its core, the app marries financial data with the subtleties of news analysis, offering an edge in a data-driven trading environment. Its ability to parse through volumes of data and provide a sentiment score is what makes it captivating.

## Journey to Deployment

### Initial Steps
- **Development**: Started with crafting the app in Python, ensuring it ran flawlessly locally.
- **Testing**: Rigorously tested functionalities to guarantee accurate sentiment analysis.

### Containerization with Docker
- **Docker Image**: Created a Docker image to containerize the application, allowing for consistent deployment across various environments.
- **Local Testing**: Ensured the Docker image was stable and operational on a local machine.

### Orchestration and Scaling
- **Docker-Compose**: Composed a Docker-Compose file to simulate multi-container deployment, preparing for a microservices architecture.
- **Kubernetes**: Authored a Kubernetes manifest to manage the app in a more scalable and resilient fashion.

### Cloud Deployment
- **Google Cloud Run**: Chose Google Cloud Run for its seamless integration and ease of use, pushing the Docker image to the cloud.
- **Live URL**: The application was deployed successfully and is now accessible [here](https://stockapp-r7agec3nya-nw.a.run.app).

## How It Works
1. Fetches real-time stock data and recent related news articles.
2. Uses OpenAI GPT-4 for nuanced sentiment analysis.
3. Provides an interface to interact with and visualize the sentiment data.

## Final Thoughts
This journey from a simple Python script to a full-fledged cloud application encapsulates the modern development lifecycle, demonstrating the power of cloud-native technologies coupled with AI.

To explore the app's functionalities, visit the documentation [here](https://stockapp-r7agec3nya-nw.a.run.app/docs#/).
