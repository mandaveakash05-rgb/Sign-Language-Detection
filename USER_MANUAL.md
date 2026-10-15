# Hand Sign Detection & Message Translation - User Manual

Welcome to the **Hand Sign Detection & Message Translation System**! This manual explains how the real-time detection works, which hand gestures map to which messages, step-by-step instructions for forming each sign, and how to use the interactive interface.

---

## 1. Quick Start

Run the detection application directly:
```powershell
python "Sign language detection.py"
```
*(or `python app.py`)*

Press **Q** or **Esc** anytime to exit.

---

## 2. Sign-to-Message Translation Reference Table

Each hand sign corresponds to a distinct, meaningful message:

| Sign Code | Gesture Name | Distinct Message / Meaning | Hand Description & Action |
| :---: | :--- | :--- | :--- |
| **HELLO** | **Open Palm (5)** | ✋ **"Hello / Welcome / Stop"** | Hold your hand up with all 5 fingers spread out facing the camera. |
| **PEACE** | **Peace / V Sign** | ✌️ **"Peace & Good Health"** | Extend your Index and Middle fingers upward in a 'V' shape; fold thumb, ring, and pinky into your palm. |
| **YES** | **Thumbs Up** | 👍 **"Yes / Approved / Good"** | Form a fist and point your thumb straight up toward the ceiling. |
| **NO** | **Thumbs Down** | 👎 **"No / Disagree / Rejected"** | Form a fist and point your thumb downward toward the floor. |
| **LOVE** | **I Love You (ILY)** | 🤟 **"I Love You"** | Extend your Thumb, Index, and Pinky fingers outward; fold your middle and ring fingers flat against your palm. |
| **OK** | **OK Sign** | 👌 **"Everything is OK / Perfect"** | Touch the tips of your Thumb and Index finger together to form a ring; keep your other 3 fingers extended upward. |
| **ATTENTION** | **Pointing (Index)** | ☝️ **"Look Here / Pay Attention"** | Point your Index finger straight up; fold your thumb, middle, ring, and pinky into a fist. |
| **CALL** | **Call Me / Shaka** | 🤙 **"Call Me / Relax / Stay Cool"** | Extend your Thumb and Pinky fingers outward; keep your middle 3 fingers folded into your palm. |
| **STRENGTH** | **Fist (Letter A)** | ✊ **"Stay Strong / Power / Solidarity"** | Curl all 4 fingers tightly into your palm with your thumb resting against the side of your index finger. |
| **LUCK** | **Together (U Sign)** | 🤞 **"Good Luck / Together"** | Extend your Index and Middle fingers straight up side-by-side touching together (no gap); curl other fingers in. |
| **CALM** | **4 Fingers Flat (B)** | 🖐️ **"Be Calm / Thank You / Quiet"** | Hold all 4 fingers (Index, Middle, Ring, Pinky) straight up touching together; tuck your thumb across your palm. |
| **WONDERFUL** | **3 Fingers Up (W)** | 🤟 **"Wonderful / Great / Number Three"** | Extend your Index, Middle, and Ring fingers upward with spaces in between; hold your pinky down with your thumb. |
| **PROMISE** | **Pinky Up (Letter I)** | 🤙 **"Promise / Excuse Me"** | Extend your Pinky finger straight up; fold your thumb, index, middle, and ring fingers tightly into your palm. |
| **LOOK** | **L Sign** | 👈 **"Look / Directing Attention"** | Extend your Index finger straight up and your Thumb outward at a 90° right angle, forming an 'L' shape. |

---

## 3. How to Perform Each Sign Correctly

### ✋ 1. Hello / Open Palm (5)
- **Position**: Open your entire hand facing the camera.
- **Tip**: Keep your fingers comfortably separated so all 5 fingers are clearly visible.

### ✌️ 2. Peace & Good Health (V Sign)
- **Position**: Raise your index and middle fingers in a "V" shape.
- **Tip**: Keep your ring and pinky fingers folded down with your thumb resting over them.

### 👍 3. Yes / Approved (Thumbs Up)
- **Position**: Make a fist and extend your thumb straight upward.
- **Tip**: Keep the other 4 fingers firmly curled into your palm.

### 👎 4. No / Disagree (Thumbs Down)
- **Position**: Make a fist and point your thumb downward.
- **Tip**: Rotate your wrist so your thumb points toward the bottom of the screen.

### 🤟 5. I Love You (ILY Sign)
- **Position**: Extend your thumb, index finger, and pinky finger simultaneously.
- **Tip**: The middle and ring fingers stay curled into your palm. This is the universal sign for "I Love You".

### 👌 6. Everything is OK (OK Sign)
- **Position**: Connect the tip of your thumb with the tip of your index finger to make a circular loop.
- **Tip**: Keep the middle, ring, and pinky fingers pointing upright.

### ☝️ 7. Look Here / Attention (Pointing)
- **Position**: Point your index finger upward.
- **Tip**: Curl your thumb, middle, ring, and pinky fingers into your palm.

### 🤙 8. Call Me / Relax (Shaka / Y Sign)
- **Position**: Extend your thumb and pinky outwards in opposite directions.
- **Tip**: Fold your index, middle, and ring fingers into your palm.

### ✊ 9. Stay Strong (Fist / Letter A)
- **Position**: Close all your fingers tightly into a ball.
- **Tip**: Rest your thumb against the side of your index finger.

### 🤞 10. Good Luck (U Sign)
- **Position**: Hold your index and middle fingers straight up touching together without any gap.
- **Tip**: Unlike the Peace sign which is spread apart, the U sign has the two fingers touching.

---

## 4. On-Screen Interface Guide

When you run the application, the window is divided into three sections:

```
+----------------------------------------------------+-----------------------------+
|                                                    |                             |
|               LIVE CAMERA FEED                     |   ASL GUIDE & INSTRUCTIONS  |
|                                                    |                             |
|  [21-Joint Hand Skeleton & Visual HUD Overlay]     |  - Live Status Card         |
|  Top Banner:                                       |  - Sign Reference Guide     |
|  - GESTURE: PEACE (V)                              |    (Row lights up green     |
|  - MESSAGE: "Peace & Good Health"                  |     when sign is active!)   |
|                                                    |  - Keyboard Shortcuts       |
|                                                    |                             |
+----------------------------------------------------+-----------------------------+
|  TRANSLATED TEXT: HELLO PEACE_                                                   |
|  [Hold sign 1s to auto-type] | [Space] Space | [Backspace] Delete | [R] Clear    |
+----------------------------------------------------------------------------------+
```

### 1. Left Section: Live Camera & Tracking Feed
- **21-Joint 3D Hand Skeleton**: Displays joints and connecting bones live on your hand.
- **Hand Capture Box**: Outlines the detected hand so you can check that the full sign is visible.
- **Top Result Banner**: Shows the detected sign, letter, and associated message in high-contrast text.
- **Thumbnail Inset**: Shows the normalized silhouette image in the top-right corner.
- **FPS Indicator**: Shows real-time processing speed (typically 30–60 FPS).

### 2. Right Section: Interactive Instruction Panel
- **Status Indicator**: Shows `HAND DETECTED` (Green) or `SEARCHING FOR HAND` (Orange).
- **Interactive Highlighting**: When you perform a sign, the matching row in the list **lights up in emerald green** with `>> [V] Peace & Good Health`, confirming your gesture immediately.
- **Cheat Sheet**: Quick reference for finger positions and actions.

### 3. Bottom Section: Translated Sentence Bar
- **Auto-Typing**: When you hold any sign steadily for $\approx 0.8$ seconds, the letter/message is automatically appended to your sentence!
- A progress bar visually shows typing accumulation.

---

## 5. Interactive Keyboard Shortcuts

| Shortcut | Description |
| :--- | :--- |
| **Q** or **Esc** | Quit and close the application |
| **S** | Save a full-resolution screenshot (`screenshot_<timestamp>.png`) |
| **P** | Save a crop of the detected hand (`hand_capture_<timestamp>_<sign>.png`) and print its result |
| **H** | Toggle the Instruction Panel sidebar on/off |
| **M** | Toggle binary segmentation mask view (for lighting diagnostics) |
| **Space** | Insert a space character in the translated text |
| **Backspace** | Delete the last typed letter from the translated text |
| **R** or **C** | Reset and clear the entire translated text buffer |

---

## 6. Tips for Optimal Hand Detection

1. **Camera Distance**: Sit about 0.5 to 1.5 meters from the webcam so your entire hand is clearly visible.
2. **Lighting**: Ensure your hand is reasonably well-lit from the front or above. Avoid very strong backlights (e.g. sitting directly in front of a bright window).
3. **Hand Orientation**: Show your palm facing toward the camera for standard gestures, and side-profile for thumbs up/down.
4. **Steady Gestures**: Hold the gesture steadily for ~0.8 seconds if you wish to auto-type the letter into the sentence bar.
5. Press **P** to save the currently outlined hand crop; the sign and message result are shown in the camera banner and printed with the capture filename.

Recognition is an estimate for the supported static gestures, not a guarantee of perfect ASL interpretation. Moving signs and signs outside the built-in set are not recognized.
