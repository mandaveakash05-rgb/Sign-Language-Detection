import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import (
    ASL_LABELS,
    DATASET_LABEL_CODES,
    IMAGE_SIDE,
    PIXEL_COUNT,
    crop_hand_region,
    load_dataset,
    load_model,
    prepare_camera_roi,
    train_model,
)


class DatasetTests(unittest.TestCase):
    def test_loads_sign_language_mnist_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "sign_mnist_train.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as dataset_file:
                writer = csv.writer(dataset_file)
                writer.writerow(["label", *(f"pixel{i}" for i in range(PIXEL_COUNT))])
                writer.writerow([0, *([0] * PIXEL_COUNT)])
                writer.writerow([24, *([255] * PIXEL_COUNT)])

            images, labels = load_dataset(dataset_path)

        self.assertEqual(images.shape, (2, PIXEL_COUNT))
        self.assertEqual(labels.tolist(), [0, 23])
        self.assertEqual(ASL_LABELS[0], "A")
        self.assertEqual(ASL_LABELS[23], "Y")

    def test_rejects_invalid_label(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "bad.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as dataset_file:
                writer = csv.writer(dataset_file)
                writer.writerow(["label", *(f"pixel{i}" for i in range(PIXEL_COUNT))])
                writer.writerow([25, *([0] * PIXEL_COUNT)])

            with self.assertRaisesRegex(ValueError, "label code 25"):
                load_dataset(dataset_path)

    def test_rejects_excluded_j_label_code(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "bad.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as dataset_file:
                writer = csv.writer(dataset_file)
                writer.writerow(["label", *(f"pixel{i}" for i in range(PIXEL_COUNT))])
                writer.writerow([9, *([0] * PIXEL_COUNT)])

            with self.assertRaisesRegex(ValueError, "code 9/J is excluded"):
                load_dataset(dataset_path)

    def test_training_saves_a_loadable_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "datasets"
            data_dir.mkdir()
            dataset_path = data_dir / "sign_mnist_train.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as dataset_file:
                writer = csv.writer(dataset_file)
                writer.writerow(["label", *(f"pixel{i}" for i in range(PIXEL_COUNT))])
                for index, label_code in enumerate(DATASET_LABEL_CODES):
                    pixels = [0] * PIXEL_COUNT
                    pixels[index] = 255
                    writer.writerow([label_code, *pixels])

            model_path = Path(temporary_directory) / "models" / "classifier.joblib"
            train_model(data_dir, model_path)

            classifier = load_model(model_path)

        self.assertEqual(len(classifier.classes_), len(ASL_LABELS))


class WebcamPreprocessingTests(unittest.TestCase):
    def test_crop_hand_region_clips_bounds_and_copies_frame(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[5:15, 10:20] = 255

        crop = crop_hand_region(frame, (8, 3, 22, 17))
        crop[:] = 0

        self.assertEqual(crop.shape, (14, 14, 3))
        self.assertEqual(int(frame[5, 10, 0]), 255)

    def test_crop_hand_region_rejects_bounds_outside_frame(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "do not overlap"):
            crop_hand_region(frame, (40, 40, 50, 50))

    def test_blank_camera_crop_has_no_hand(self):
        self.assertIsNone(prepare_camera_roi(np.zeros((160, 160, 3), dtype=np.uint8)))

    def test_finds_a_contrasting_hand_shape(self):
        roi = np.zeros((160, 160, 3), dtype=np.uint8)
        roi[35:130, 65:100] = 255
        image = prepare_camera_roi(roi)

        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (IMAGE_SIDE, IMAGE_SIDE))
        self.assertGreater(int(image.max()), 0)


class CommandLineAndModelTests(unittest.TestCase):
    def test_ensure_default_model_creates_valid_model(self):
        from app import ensure_default_model

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "default.joblib"
            ensure_default_model(model_path)
            self.assertTrue(model_path.is_file())
            clf = load_model(model_path)
            self.assertEqual(len(clf.classes_), len(ASL_LABELS))

    def test_build_parser_defaults_command_to_none(self):
        from app import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.command)


class InstructionPanelAndGestureTests(unittest.TestCase):
    def test_analyze_hand_contour_empty_roi_raises(self):
        from app import analyze_hand_contour

        with self.assertRaises(ValueError):
            analyze_hand_contour(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_analyze_hand_contour_blank_roi(self):
        from app import analyze_hand_contour

        canvas, cnt, gesture, letter, fingers, pc, pr, tips, defs, th = analyze_hand_contour(
            np.zeros((160, 160, 3), dtype=np.uint8)
        )
        self.assertIsNone(canvas)
        self.assertIsNone(cnt)
        self.assertEqual(gesture, "No Hand")

    def test_analyze_hand_contour_contrasting_shape(self):
        from app import analyze_hand_contour

        roi = np.zeros((160, 160, 3), dtype=np.uint8)
        roi[35:130, 65:100] = 255
        canvas, cnt, gesture, letter, fingers, pc, pr, tips, defs, th = analyze_hand_contour(roi)
        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape, (IMAGE_SIDE, IMAGE_SIDE))
        self.assertIsNotNone(cnt)
        self.assertIn(letter, "ABCDUVWYZ5")

    def test_analyze_hand_contour_self_intersecting_contour_does_not_crash(self):
        from app import analyze_hand_contour

        roi = np.zeros((160, 160, 3), dtype=np.uint8)
        points = np.array([
            [40, 40], [80, 90], [40, 120], [120, 80], [80, 40], [120, 120],
            [60, 20], [20, 80], [60, 140], [100, 80], [20, 20], [140, 80]
        ], dtype=np.int32)
        cv2.fillPoly(roi, [points], 255)

        canvas, cnt, gesture, letter, fingers, pc, pr, tips, defs, th = analyze_hand_contour(roi)
        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape, (IMAGE_SIDE, IMAGE_SIDE))
        self.assertIn(letter, "ABCDUVWYZ5")

    def test_draw_instruction_panel_renders(self):
        from app import draw_instruction_panel

        panel = np.zeros((560, 380, 3), dtype=np.uint8)
        draw_instruction_panel(panel, active_code="PEACE", hand_detected=True, gesture_name="Peace / V Sign", message_text="Peace & Good Health")
        self.assertGreater(int(panel.max()), 0)

    def test_draw_sentence_bar_renders(self):
        from app import draw_sentence_bar

        bar = np.zeros((75, 1020, 3), dtype=np.uint8)
        draw_sentence_bar(bar, sentence="HELLO", typing_item="PEACE", progress=0.6)
        self.assertGreater(int(bar.max()), 0)

    def test_word_suggestions_match_spelled_prefixes(self):
        from app import suggest_words

        self.assertIn("HI", suggest_words("H"))
        self.assertIn("BYE", suggest_words("BY"))
        self.assertIn("LIKE", suggest_words("LIK"))
        self.assertEqual(suggest_words(""), ())

    def test_completes_current_word_without_changing_previous_words(self):
        from app import complete_current_word

        self.assertEqual(complete_current_word("H"), "HI")
        self.assertEqual(complete_current_word("HI BY"), "HI BYE")
        self.assertEqual(complete_current_word("HI "), "HI ")
        self.assertEqual(complete_current_word("XYZ"), "XYZ")


if __name__ == "__main__":
    unittest.main()
