<!-- PROJECT LOGO -->
<br />
<p align="center">
    <!-- <img src="" alt="Screenshot" width="80" height="80"> -->

  <h3 align="center">AI Chatbot</h3>

  <p align="center">
    A simple chatbot powered by Pytorch & NPL.
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    · -->
    <a href="https://github.com/mindninjaX/AI-Chatbot/issues">Report Bug</a>
    ·
    <a href="https://github.com/mindninjaX/AI-Chatbot/issues">Request Feature</a>
  </p>
</p>
<br />
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#customize">Customize</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <!-- <li><a href="#license">License</a></li> -->
    <li><a href="#contact">Contact</a></li>
    <li><a href="#resources">Resources</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

![carbon](https://user-images.githubusercontent.com/59148052/106388679-b50bde00-6405-11eb-80d8-b1990000de06.png)


- The implementation should be easy to follow for beginners and provide a basic understanding of chatbots.
- The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.
Customization for your own use case is super easy. Just modify `intents.json` with possible patterns and responses and re-run the training.

### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

- [Pytorch](https://pytorch.org/)
- [Natural Language Toolkit](https://www.nltk.org/_modules/nltk/util.html)
- [Numpy](https://numpy.org/)

<!-- GETTING STARTED -->

## Getting Started

To get started with this project, follow the instructions below:

### Prerequisites

To run and work with this project you need to have the latest version of Python installed in your system.

Along with Python, we would also need some python modules to work with this project. Check <a href="#installation">Installation</a> for instructions on same.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/mindninjaX/AI-Chatbot
   ```
3. Install required packages using `pip install`
   ```sh
   pip install torch
   pip install numpy
   pip install nltk
   ```
4. Uncomment `nltk.download()` & run **nltk_utlis.py** to install _Punkt Tokenzier model_
   ```Python
   import nltk
   nltk.download()
   ```

## Usage
Run `train.py` to train the chatbot
  ```sh
  python train.py
  ```
This is create a new file `data.pth` which contains the trained data for our chatbot.

Initiate the chatbot! Run `chat.py`
  ```sh
  python chat.py
  ```

## Customize
Raw data is present in `intents.json`. Customize this file as per your needs. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. **You have to re-run the training whenever this file is modified.**
  ```json
  {
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
  ```

## Roadmap

See the [open issues](https://github.com/mindninjaX/AI-Chatbot/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->

## Contact

### Rishabh Singh
- **Twitter** -  [@mindninjaX](https://twitter.com/mindninjaX)
- **LinkedIn** - [linkedin/mindninjax](https://www.linkedin.com/in/mindninjax/)
- **Email** - rishabh.singh@studentambassadors.com

**Project Link:** [https://github.com/mindninjaX/AI-Chatbot](https://github.com/mindninjaX/AI-Chatbot)

<!-- ACKNOWLEDGEMENTS -->

## Resources & Acknowledgements

- [VS Code](https://code.visualstudio.com/)
- [Python Setup](https://www.python.org/downloads/)
- [Python Documentation](https://docs.python.org/)
- [NLTK Documentation](https://www.nltk.org/_modules/nltk/util.html)
- [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
- [JSON Guide](https://www.json.org/)
