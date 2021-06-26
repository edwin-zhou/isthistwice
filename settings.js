var config = {
    BATCH_SIZE: 45,
    BATCHES_PER_EPOCH: 30,
    EPOCHS: 50,
    VALIDATION_SIZE: 108,
    VALIDATION_BATCH_SIZE: 27,
    IMG_SIZE: [256,256],
    IMG_PATH: 'images/processed',
    LABELS: ['Tzuyu', 'Chaeyoung', 'Dahyun', 'Mina', 'Jihyo', 'Sana', 'Momo', 'Jeongyeon', 'Nayeon'],
    MODEL_NAME: 'ot9'
}

module.exports = config