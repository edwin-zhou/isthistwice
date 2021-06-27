var config = {
    BATCH_SIZE: 54,
    BATCHES_PER_EPOCH: 25,
    EPOCHS: 50,
    VALIDATION_SIZE: 54,
    VALIDATION_BATCH_SIZE: 9,
    IMG_SIZE: [256,256],
    IMG_PATH: 'images/processed',
    LABELS: ['Tzuyu', 'Chaeyoung', 'Dahyun', 'Mina', 'Jihyo', 'Sana', 'Momo', 'Jeongyeon', 'Nayeon'],
    MODEL_NAME: 'ot9'
}

module.exports = config