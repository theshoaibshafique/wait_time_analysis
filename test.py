from inference import get_model


def main():
    modelId = 'coco/24'
    model = get_model(model_id=modelId)


if __name__ == '__main__':
    main()