# shortest path in a multi-linestring

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Requirements](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project
How to find the shortest path between two linestrings of a multi-linestring.

### Requirements
* Docker

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* Install Docker

### Installation
 
1. Clone the repo
```sh
git clone https://github.com/tachillon/shortest-path-in-a-multi-linestring.git
```
2. Build the docker
```sh
cd shortest-path-in-a-multi-linestring && docker build -t dikjstra:1 .
```
<!-- USAGE EXAMPLES -->
## Usage
```sh
docker run --rm -it -v ${PWD}:/tmp dikjstra:1 python3 dikjstra.py
```


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Achille-Tâm GUILCHARD - achilletamguilchard@gmail.com
