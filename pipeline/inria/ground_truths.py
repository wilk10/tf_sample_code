import cv2
import pandas
import numpy as np
from .image_preparation import ImagePreparation


class InriaGroundTruths:
    EXTENSION = "csv"
    COLUMNS = ["filename", "split_id", "width", "height", "class", "label", "xmin", "ymin", "xmax", "ymax"]
    CONDITIONS_COLUMNS = ["within_margin", "area_is_ok", "dims_are_ok"]
    MARGIN = 5
    LOWER_AREA_LIMIT = 1000
    UPPER_AREA_LIMIT = 30000
    MIN_DIM_SIZE = 33

    def __init__(self, width_height_dir, tif_filename):
        self.width_height_dir = width_height_dir
        self.tif_filename = tif_filename
        self.image_name = self.tif_filename.split(".")[0]
        self.png_filename = "{}.png".format(self.image_name)
        self.train_dir = self.width_height_dir.parent.parent
        self.images_tif_dir = self.train_dir / "images"
        self.images_png_dir = self.train_dir / "images_png"
        self.ground_truth_dir = self.train_dir / "gt"
        self.width_height = self.width_height_dir.name
        self.csv_filename = "{}.{}".format(self.image_name, self.EXTENSION)
        self.base_dim = int(self.width_height.split("x")[0])
        self.coordinate_columns = ["xmin", "ymin", "xmax", "ymax"]

    def make_target_directory_by_flag(self, research_type):
        split_images_dir = self.train_dir / "split_images"
        split_images_width_height_dir = split_images_dir / self.width_height
        if not split_images_width_height_dir.exists():
            split_images_width_height_dir.mkdir()
        roofs_dir = "single_roofs" if research_type == "instance_segmentation" else "resized_single_roofs"
        single_roofs_dir = self.train_dir / roofs_dir
        directory_by_flag = {
            "split": split_images_width_height_dir,
            "colour": single_roofs_dir / "colour",
            "gt": single_roofs_dir / "gt",
            "cropped_gt": single_roofs_dir / "cropped_gt"}
        return directory_by_flag

    @staticmethod
    def find_top_bottom_corners(contour):
        contour = contour.reshape(-1, 2)
        axes = [0, 1]
        top_left = [None, None]
        bottom_right = [None, None]
        for axis in axes:
            min_id = np.argmin(contour[:, axis])
            top_left[axis] = contour[min_id][axis]
            max_id = np.argmax(contour[:, axis])
            bottom_right[axis] = contour[max_id][axis]
        return [tuple(top_left), tuple(bottom_right)]

    @staticmethod
    def make_xy_in_flat_box_to_tuples(flat_boxes, invert_xy=False):
        tupled_xys = []
        for flat_box in flat_boxes:
            reshaped_xy = np.reshape(flat_box, (2, 2))
            if invert_xy:
                tuple_xy = [(xy[1], xy[0]) for xy in reshaped_xy]
            else:
                tuple_xy = [(xy[0], xy[1]) for xy in reshaped_xy]
            tupled_xys.append(tuple_xy)
        return tupled_xys

    @staticmethod
    def add_boxes(image, boxes):
        for box in boxes:
            (x1, y1), (x2, y2) = box
            if isinstance(x1, np.float64):
                assert 0 <= x1 <= 1
                height, width, channels = image.shape
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    @classmethod
    def show_images(cls, image1, image2=None, df=None, boxes=None, contours=None):
        if not boxes:
            assert df is not None
            last_columns = ["xmin", "ymin", "xmax", "ymax"]
            flat_boxes = df.loc[:, last_columns].values.tolist()
            boxes = cls.make_xy_in_flat_box_to_tuples(flat_boxes)
        if image2 is None:
            image2 = image1.copy()
            if contours:
                for contour in contours:
                    for (x, y) in contour:
                        cv2.circle(image1, (x, y), 1, (255, 0, 0), 3)
        else:
            assert not contours
            cls.add_boxes(image1, boxes)
        cls.add_boxes(image2, boxes)
        out = np.hstack([image1, image2])
        cv2.imshow('Output', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_boxes(self, image):
        assert image is not None
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                box = self.find_top_bottom_corners(contours[i])
                (x1, y1), (x2, y2) = box
                if x2 >= x1 and y2 >= y1:
                    boxes.append(box)
        return np.array(boxes)

    def load_full_image(self, flag):
        assert flag in ["colour", "gt"]
        if flag == "gt":
            filepath = self.ground_truth_dir / self.tif_filename
        else:
            png_filepath = self.images_png_dir / self.png_filename
            if not png_filepath.exists():
                tif_filepath = self.images_tif_dir / self.tif_filename
                image = cv2.imread(str(tif_filepath), cv2.IMREAD_COLOR)
                cv2.imwrite(str(png_filepath), image)
            filepath = png_filepath
        return cv2.imread(str(filepath), cv2.IMREAD_COLOR)

    def get_df_with_image_data(self, image, df=None, split_id=None):
        if df is None:
            df = pandas.DataFrame(columns=self.COLUMNS)
        row = pandas.Series(index=self.COLUMNS)
        row["filename"] = self.png_filename
        row["split_id"] = split_id if split_id is not None else np.nan
        row["width"] = image.shape[1]
        row["height"] = image.shape[0]
        boxes = self.find_boxes(image)
        for box in boxes:
            row["class"] = "roof"
            row["label"] = 1
            (x1, y1), (x2, y2) = box
            relative_box = [
                x1 / row["width"], y1 / row["height"],
                x2 / row["width"], y2 / row["height"]]
            row.loc[self.coordinate_columns] = relative_box
            df = df.append(row, ignore_index=True)
        if len(boxes) == 0:
            df = df.append(row, ignore_index=True)
        return df

    def get_csv_filepath_by_flag(self, flag):
        csv_dir_by_flag = {
            "object_detection": self.width_height_dir,
            "instance_segmentation": self.width_height_dir.parent / "full_images",
            "custom_keras": self.width_height_dir.parent / "full_images"}
        return csv_dir_by_flag[flag] / self.csv_filename

    def get_df_object_detection(self):
        csv_filepath = self.get_csv_filepath_by_flag("object_detection")
        if csv_filepath.exists():
            df = pandas.read_csv(csv_filepath, index_col=0)
        else:
            df = pandas.DataFrame(columns=self.COLUMNS)
            image_gt = self.load_full_image("gt")
            split_images = ImagePreparation.split(self.base_dim, image_gt)
            for split_id, split_image in enumerate(split_images):
                df = self.get_df_with_image_data(split_image, df, split_id)
            df.to_csv(csv_filepath)
        return df

    def get_df_instance_segmentation(self):
        csv_filepath = self.get_csv_filepath_by_flag("instance_segmentation")
        if csv_filepath.exists():
            df = pandas.read_csv(csv_filepath, index_col=0)
        else:
            image_gt = self.load_full_image("gt")
            df = self.get_df_with_image_data(image_gt)
            df = self.check_conditions(df)
            df.to_csv(csv_filepath)
        return df

    def convert_from_df_row_to_coordinate_box(self, row):
        normalized_coordinates = row[self.coordinate_columns]
        x1 = round(normalized_coordinates[0] * row["width"])
        y1 = round(normalized_coordinates[1] * row["height"])
        x2 = round(normalized_coordinates[2] * row["width"])
        y2 = round(normalized_coordinates[3] * row["height"])
        return [(x1, y1), (x2, y2)]

    def check_conditions(self, df):
        within_margin_conditions = []
        area_is_ok_conditions = []
        dims_are_ok_conditions = []
        isolated_filenames = []
        for i, image_row in df.iterrows():
            box = self.convert_from_df_row_to_coordinate_box(image_row)
            (x1, y1), (x2, y2) = box
            box_width = (x2 - x1)
            box_height = (y2 - y1)
            area = box_width * box_height
            beyond_margins = self.check_box_beyond_margins(
                box, image_row["height"], image_row["width"], margin=self.MARGIN, beyond_all=False)
            within_margin_conditions.append(not beyond_margins)
            area_is_ok = self.LOWER_AREA_LIMIT < area < self.UPPER_AREA_LIMIT
            area_is_ok_conditions.append(area_is_ok)
            dims_are_ok = box_height >= self.MIN_DIM_SIZE and box_width >= self.MIN_DIM_SIZE
            dims_are_ok_conditions.append(dims_are_ok)
            isolated_filename = "{}_{}.png".format(self.image_name, i)
            isolated_filenames.append(isolated_filename)
        df["within_margin"] = within_margin_conditions
        df["area_is_ok"] = area_is_ok_conditions
        df["dims_are_ok"] = dims_are_ok_conditions
        df["isolated_filename"] = isolated_filenames
        df["isolated_roof"] = np.nan
        return df

    def get_split_or_single_image_filepath(self, image_id, flag, research_type):
        target_directory_by_flag = self.make_target_directory_by_flag(research_type)
        directory = target_directory_by_flag[flag]
        target_filename = "{}_{}.png".format(self.image_name, image_id)
        return directory / target_filename

    @staticmethod
    def check_box_beyond_margins(box, height, width, margin=0, beyond_all=False):
        (x1, y1), (x2, y2) = box
        on_left = x1 <= margin
        on_top = y1 <= margin
        on_right = x2 >= width - margin - 1
        on_bottom = y2 >= height - margin - 1
        if beyond_all:
            beyond_margins = all([on_left, on_top, on_right, on_bottom])
        else:
            beyond_margins = any([on_left, on_top, on_right, on_bottom])
        return beyond_margins

    # todo: this comes from RoofEdges. i need to move image manipulation methods somewhere central easily accessible
    @staticmethod
    def find_contours(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    def add_isolated_roof_flag_and_save_images(self, df, research_type):
        result_df = df.copy()
        for column in self.CONDITIONS_COLUMNS:
            df = df[df[column] == True]
        image_colour = self.load_full_image("colour")
        image_gt = self.load_full_image("gt")
        for i, image_row in df.iterrows():
            box = self.convert_from_df_row_to_coordinate_box(image_row)
            roof_image_colour = ImagePreparation.crop_on_box(image_colour, box, self.MARGIN)
            roof_image_gt = ImagePreparation.crop_on_box(image_gt, box, self.MARGIN)
            output_image_gt = roof_image_gt.copy()
            contours, hierarchy = self.find_contours(roof_image_gt)
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            max_area = max(contour_areas)
            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area == max_area:
                    cv2.drawContours(output_image_gt, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
                elif hierarchy[0][j][-1] == -1:
                    cv2.drawContours(output_image_gt, [contour], 0, (0, 0, 0), thickness=cv2.FILLED)
            leftout_boxes = self.find_boxes(output_image_gt)
            assert len(leftout_boxes) == 1
            main_roof_box = leftout_boxes[0]
            height, width, _ = output_image_gt.shape
            beyond_margins = self.check_box_beyond_margins(main_roof_box, height, width, margin=0, beyond_all=False)
            if not beyond_margins:
                result_df.loc[i, "isolated_roof"] = True
                cropped_image_gt = ImagePreparation.crop_on_box(output_image_gt, main_roof_box)
                cropped_image_gray = cv2.cvtColor(cropped_image_gt, cv2.COLOR_BGR2GRAY)
                cropped_image_gray = np.expand_dims(cropped_image_gray, axis=3)
                if research_type == "instance_segmentation":
                    image_by_flag = {
                        "colour": roof_image_colour, "gt": output_image_gt, "cropped_gt": cropped_image_gray}
                else:
                    resized_image_colour = ImagePreparation.resize_image(roof_image_colour, self.base_dim)
                    resized_image_gt = ImagePreparation.resize_image(output_image_gt, self.base_dim)
                    reshaped_image_gt = np.expand_dims(cv2.cvtColor(resized_image_gt, cv2.COLOR_BGR2GRAY), axis=3)
                    image_by_flag = {"colour": resized_image_colour, "gt": reshaped_image_gt}
                for flag, image in image_by_flag.items():
                    image_path = self.get_split_or_single_image_filepath(i, flag, research_type)
                    cv2.imwrite(str(image_path), image)
            else:
                result_df.loc[i, "isolated_roof"] = False
        csv_filepath = self.get_csv_filepath_by_flag(research_type)
        result_df.to_csv(csv_filepath)
        return result_df

    def match_isolated_colour_and_gt_roofs(self, research_type):
        df = self.get_df_instance_segmentation()
        if df["isolated_roof"].isnull().values.all():
            df = self.add_isolated_roof_flag_and_save_images(df, research_type)
        isolated_roofs = []
        isolated_df = df[df["isolated_roof"] == True]
        for i, row in isolated_df.iterrows():
            isolated_roof = {
                "colour_image_path": self.get_split_or_single_image_filepath(i, "colour", research_type),
                "gt_image_path": self.get_split_or_single_image_filepath(i, "gt", research_type),
                "cropped_gt_image_path": self.get_split_or_single_image_filepath(i, "cropped_gt", research_type)}
            isolated_roofs.append(isolated_roof)
        return isolated_roofs
