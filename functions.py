import csv
import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        # Write header names to csv file
        header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                  'license_number_score']
        f.write(','.join(header) + '\n')

        for frame_nmr, frame_results in results.items():
            for car_id, car_info in frame_results.items():
                car_bbox = tuple(car_info.get('car', {}).get('bbox', []))  # Convert to tuple
                license_plate_info = car_info.get('license_plate', {})
                license_plate_bbox = tuple(license_plate_info.get('bbox', []))  # Convert to tuple
                license_plate_bbox_score = car_info.get('bbox_score', '')
                license_number = car_info.get('text', '')
                license_number_score = car_info.get('text_score', '')

                # Convert bbox tuples to formatted strings
                car_bbox_str = '[{}]'.format(' '.join(map(str, car_bbox)))
                license_plate_bbox_str = '[{}]'.format(' '.join(map(str, license_plate_bbox)))

                # Write row to CSV
                row = [frame_nmr, car_id, car_bbox_str, license_plate_bbox_str, license_plate_bbox_score,
                       license_number, license_number_score]
                f.write(','.join(map(str, row)) + '\n')

    print("CSV file written successfully.")


def write_cars_to_csv(cars, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['frame_nmr', 'car_id', 'car_bbox']
        writer.writerow(header)

        for frame_nmr, cars_dict in cars.items():
            for car_id, car_info in cars_dict.items():
                car_bbox = tuple(car_info.get('car', {}).get('bbox', []))
                car_bbox_str = '({})'.format(' '.join(map(str, car_bbox)).replace(' ', ','))

                row = [frame_nmr, car_id, car_bbox_str]
                writer.writerow(row)

    print("CSV file written successfully.")


def license_complies_format_UK(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format_UK(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    found = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_index = j
            found = True
            break

    if found:
        return vehicle_track_ids[car_index]

    return -1, -1, -1, -1, -1
