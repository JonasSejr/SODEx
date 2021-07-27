import numpy as np
from lime.lime_image import LimeImageExplainer

from src.data_model import Image, PascalVOCObject


class Explainer:

    def __init__(self, logger, object_extractor):
        """
        :param logger: Standart python logger
        :param object_extractor: An object extractor with a method get_bboxes.
        The method should take an image as an array and return a list of bounding boxes detected in the image.
        Additionally object detector needs to supply a method bboxes_iou.
        bboxes_iou must take two lists of bboxes and calculate the pairwise intersection of union between the bounding
        boxes
        """
        self.logger = logger
        self.object_extractor = object_extractor

    def _get_class_probability_predictor(self, voc_object: PascalVOCObject):
        def predict_class_probability(images_as_arrays):
            scores = []
            for image_as_array in images_as_arrays:
                bboxes = np.array(self.object_extractor.get_bboxes(image_as_array))
                if (len(bboxes) == 0):
                    type_probability = 0  # Probability 0 if no objects are detected.
                    self.logger.info("Object not found when explaining: " + voc_object.image.image_path.stem)
                else:
                    ious = self.object_extractor.bboxes_iou(voc_object.bbox[np.newaxis, :4], bboxes[:, :4])
                    filtered_ious = ious[ious > 0.4]
                    if(len(filtered_ious) == 0):
                        type_probability = 0
                    else:
                        max_ix = np.argmax(ious)
                        type_probability = bboxes[max_ix][4]
                scores.append(np.array([type_probability, 1 - type_probability]))
            return np.array(scores)

        return predict_class_probability

    def get_class_probability_explanation(self, object: PascalVOCObject):
        explainer = LimeImageExplainer(verbose=True)
        self.logger.info("Explaining object: ")
        explanation = explainer.explain_instance(
            image=object.image.as_array(),
            classifier_fn=self._get_class_probability_predictor(object),
            num_samples=1000)
        return explanation
