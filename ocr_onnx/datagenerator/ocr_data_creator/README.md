# OCR DATA CREATOR

# Code

# Usage
'''
python image_creator.py --num_image=1000
'''

'''
labeling example

"info": {},
	"license": {},
	"images": [
		{
			"id": "1583398098311",
			"width": 512,
			"height": 512,
			"file_name": "1583398098311-0.jpg",
			"data_captured": 1583398099.109605
		}
	],
	"annotations": [
        {
            "id": "1583398098311",
			"text": [
				{
					"text_id": "1583398098311-0-0",
					"contents": "큼" 
				},
				{
					"text_id": "1583398098311-0-1",
					"contents": "맑"
				},
            ]
            "bbox": [
				{
					"bbox_id": "1583398098311-0-0",
					"contents": [
						165, 
						19,
						25,
						27
					]
				},
				{
					"bbox_id": "1583398098311-0-1",
					"contents": [
						190,
						19,
						24,
						27
					]
				},
            ]
        }
    ]

bbox contents -> [center_x, center_y, width, height]
'''
