import numpy as np

import ocr7


def test_pipeline():
    pipeline = ocr7.ocr_model.Pipeline()

    # We shouldn't find any text in a blank image.
    assert len(pipeline.recognize(images=[np.zeros((256, 256, 3), dtype='uint8')])[0]) == 0

    image = ocr7.utils.read('tests/test_image.jpg')

    # Predictions is a list of (text, box) tuples.
    predictions = pipeline.recognize(images=[image])[0]

    assert len(predictions) == 1
    assert predictions[0][0] == 'eventdock'
