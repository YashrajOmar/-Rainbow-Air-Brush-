# Rainbow Air Brush

Rainbow Air Brush is a gesture-controlled painting app that allows users to draw on a virtual canvas using only hand movements, without a mouse or stylus. Built with Python, OpenCV, and MediaPipe, it tracks hand landmarks through your webcam and lets you paint, erase, or clear the canvas in real-time.

---

## 🖌️ How It Works

* Uses your webcam to track your hand.
* When you **pinch** (thumb + index finger), it draws.
* When you **show an open palm**, it enters **eraser** mode.
* Press **C** to clear the canvas.
* Press **ESC** to exit.

---

## ⚙️ Libraries Used

| Library     | Purpose                                |
| ----------- | -------------------------------------- |
| `OpenCV`    | Webcam feed, drawing, image processing |
| `MediaPipe` | Real-time hand landmark detection      |
| `NumPy`     | Array operations and canvas management |

---

## 🚀 Installation & Setup

1. Install Python (3.6 or higher).

2. Install dependencies:

   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. Run the application:

   ```bash
   python "rainbow air brush.py"
   ```

---

## 🕹️ Commands and Controls

| Action       | How to Perform                        |
| ------------ | ------------------------------------- |
| Draw         | Pinch thumb and index finger          |
| Erase        | Show an open palm (5 fingers visible) |
| Clear Canvas | Press `C` key                         |
| Exit App     | Press `ESC` key                       |

---

## 🤖 Future Scope

* 📁 Save and load your drawings
* 🎤 Voice commands (e.g., "clear", "blue brush")
* □ Shape drawing (rectangle, circle, triangle)
* 📱 Turn into mobile/web-based app
* 🧠 AI-based gesture classification

---

## ✨ Why It's Cool

* Hands-free interaction using AI
* Fun for kids and artists alike
* Great intro project for computer vision + AI

---

Feel free to expand this project, add colors, shapes, or train your own hand gesture recognition model!

Made with ❤️ using Python, OpenCV, and MediaPipe.
