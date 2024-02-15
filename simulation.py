import cv2
import numpy as np
import pandas as pd
import ast


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def process_frame_results(frame, results_in_frame, license_plate):
    for _, row in results_in_frame.iterrows():
        car_bbox = ast.literal_eval(row['car_bbox'])
        car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)
        license_bbox = ast.literal_eval(row['license_plate_bbox'])
        x1, y1, x2, y2 = map(int, license_bbox)
        car_id = row['car_id']
        draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), color=(0, 255, 0), thickness=5, line_length_x=200,
                    line_length_y=200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)
        license_crop = license_plate[car_id]['license_crop']
        H, W, _ = license_crop.shape
        text_background = np.ones((100, 300, 3), dtype=np.uint8) * 255
        # Add text to the white background
        text = "PASSED"
        license_plate_text = license_plate[car_id]['license_plate_number']
        print("License plate text: ", license_plate_text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.0
        font_thickness = 2
        font_color_passed = (0, 255, 0)  # Green color for "PASSED"
        font_color_license = (0, 0, 0)  # Black color for license plate number
        # Calculate text size and position for "PASSED"
        (text_width_passed, text_height_passed), _ = cv2.getTextSize(text, font, font_size, font_thickness)
        text_position_passed = (
            (text_background.shape[1] - text_width_passed) // 2,
            (text_background.shape[0] + text_height_passed) // 2 - 20)
        # Calculate text size and position for license plate number
        (text_width_license, text_height_license), _ = cv2.getTextSize(license_plate[car_id]['license_plate_number'],
                                                                       font, font_size, font_thickness)
        text_position_license = ((text_background.shape[1] - text_width_license) // 2, (
                text_background.shape[0] + text_height_license) // 2 + 20)  # Adjusted for license plate position
        # Draw "PASSED" text
        cv2.putText(text_background, text, text_position_passed, font, font_size, font_color_passed, font_thickness)
        # Draw license plate number text
        cv2.putText(text_background, license_plate_text, text_position_license, font,
                    font_size, font_color_license, font_thickness)
        # Now, you can put this text region onto your original frame
        frame[int(car_bbox[1] - 50):int(car_bbox[1] + 50),
        int((car_bbox[2] + car_bbox[0] - text_background.shape[1]) / 2):
        int((car_bbox[2] + car_bbox[0] + text_background.shape[1]) / 2), :] = text_background

        # display_frame = cv2.resize(frame, (1920, 1080))
        # cv2.imshow("Text Image", display_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return frame


def process_frame_car_results(frame, car_results_in_frame, results_in_frame):
    for _, row in car_results_in_frame.iterrows():
        car_bbox = ast.literal_eval(row['car_bbox'])
        car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)
        print("car_y1: " + str(car_y1) + "car_y2:" + str(car_y2))
        car_id = row['car_id']
        car_limit = car_y2 - car_y1
        print("Before: " + str(car_limit))
        print("Car ID before: " + str(car_id))


        if int(car_id) not in results_in_frame:
            if car_limit > 150 and int(car_x2) > 1100 and int(car_y2) > 1000:
                # Add text to the white background
                text = "Manuel Control"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.6  # Adjust the font size to make the text smaller
                font_thickness = 2
                font_color = (255, 0, 0)  # Blue color for "Manuel Control"

                (text_width, text_height), _ = cv2.getTextSize(text, font, font_size, font_thickness)

                max_car_background_width = 300  # Adjust the maximum width to make the background smaller
                car_background_width = min(text_width + 20,
                                           max_car_background_width)  # Adjust the margin for a smaller background
                car_background = np.ones((60, car_background_width, 3),
                                         dtype=np.uint8) * 255  # Adjust the height for a smaller background

                text_position = (
                    (car_background.shape[1] - text_width) // 2,
                    (car_background.shape[0] + text_height) // 2
                )

                cv2.putText(car_background, text, text_position, font, font_size, font_color, font_thickness)

                car_length = car_bbox[2] - car_bbox[0]
                count_value = (car_length - car_background_width) / 2
                start_x = int(car_bbox[0] + count_value)
                end_x = int(car_bbox[0] + car_background.shape[1] + count_value)

                if start_x < 0:
                    start_x = 0

                if end_x > frame.shape[1]:
                    end_x = frame.shape[1]

                # Check if the width is still positive after adjustments
                if end_x > start_x:
                    # Place car_background in the frame
                    frame[int(car_bbox[1] - 30):int(car_bbox[1] + 30), start_x:end_x, :] = car_background
                else:
                    print("Warning: Horizontal size is zero or negative. Skipping this case.")
                # Display code (commented out for now)
                # display_frame = cv2.resize(frame, (1920, 1080))
                # cv2.imshow("Text Image", display_frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()


        else:
            pass
    return frame


def process_video(results_in_video, car_results_in_video, video_path_in_video, output_path_in_video):
    cap = cv2.VideoCapture(video_path_in_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = map(int, (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path_in_video, fourcc, fps, (width, height))

    license_plate = {}

    # Processing for 'results'
    for car_id, car_group in results_in_video.groupby('car_id'):
        car_group = car_group[car_group['license_number'] != '0']

        if not car_group.empty:
            most_repeated_license = car_group['license_number'].value_counts().idxmax()
            most_repeated_rows = car_group[car_group['license_number'] == most_repeated_license]
            sorted_group = most_repeated_rows.sort_values(by=['license_number_score'], ascending=[False])

            for _, max_score_row in sorted_group.iterrows():
                if max_score_row['license_number'] == most_repeated_license:
                    break

            frame_number = max_score_row['frame_nmr']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            _, frame = cap.read()

            license_bbox = ast.literal_eval(max_score_row['license_plate_bbox'])
            x1, y1, x2, y2 = map(int, license_bbox)
            new_width = int((x2 - x1) * 400 / (y2 - y1))
            new_height = 400

            license_crop = cv2.resize(frame[y1:y2, x1:x2, :], (new_width, new_height))

            license_plate[car_id] = {
                'license_crop': license_crop,
                'license_plate_number': max_score_row['license_number']
            }

    # Processing for 'car_results'
    for car_id, car_group in car_results_in_video.groupby('car_id'):
        frame_number = car_group['frame_nmr'].iloc[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = cap.read()

        car_bbox = ast.literal_eval(car_group['car_bbox'].iloc[0])
        x1, y1, x2, y2 = map(int, car_bbox)


    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        frame_nmr += 1

        if not ret:
            break

        # Processing for 'results'
        df_results = results_in_video[results_in_video['frame_nmr'] == frame_nmr]
        frame = process_frame_results(frame, df_results, license_plate)

        # Processing for 'car_results'
        df_car_results = car_results_in_video[car_results_in_video['frame_nmr'] == frame_nmr]
        frame = process_frame_car_results(frame, df_car_results, df_results)

        out.write(frame)

    out.release()
    cap.release()


# Example usage:
if __name__ == "__main__":
    results = pd.read_csv('plate_results.csv')
    car_results = pd.read_csv('./cars_results.csv')
    video_path = 'Traffic.mp4'
    output_path = 'Traffic_out.mp4'
    process_video(results, car_results, video_path, output_path)
