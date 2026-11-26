# Veritas smoke-test samples

These original, fictional fixtures exercise the application. The image, clip and
tone are generated media. Their scores are unsuitable for evaluating accuracy.

| Website mode | Input | Expected analyzed signals |
| --- | --- | --- |
| News | Paste `news.txt` | News credibility |
| Text | Paste `text.txt` | AI text |
| Image | Upload `image.png` | AI image, image manipulation |
| Video | Upload `video.avi` | Deepfake video, frame deletion |
| Audio | Upload `audio.wav` | Synthetic audio, partial speech editing, voice conversion |
| Combined | Paste `news.txt` and upload `image.png` | News credibility, AI text, AI image, image manipulation, title/image context |

Leave online reporting off to reproduce the local results. From the repository
root, run `python -m pytest tests -q -k public_sample` in the project virtual
environment to exercise all six cases through the API.
