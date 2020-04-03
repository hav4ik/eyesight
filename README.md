# Eyesight

Eyesight is a high-level minimalistic framework for Computer Vision on the Edge, written in Python. It was developed with focus on performance. Supports Coral Edge TPU, Pi Camera, and more.

## Design choices

*  **Python over C++.** While the later is much more efficient and more suitable for production-ready applications, the main purpose of this library is to provide a simple platform for rapid prototyping. As the majority of Computer Vision code is written in Python, it was a natural choice. It is encouraged, however, to implement your algorithms on C++ whenever possible and use this library only as a convenient wrapper.
