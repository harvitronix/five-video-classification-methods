config = {
    'datasets': {
        'ucf101': {
            'nb_classes': 101,
        },
        'synthetic_boxes': {
            'nb_classes': 5,
            'parameters': [
                'linear_move',
                'jitter_move',
                'random_move',
                'random_angle',
                'random_angle_per_frame',
                'background_shapes',
                'random_background_color',
                'random_bg_per_frame',
                'random_foreground_color',
                'random_fg_per_frame',
            ],
        },
    },
}
