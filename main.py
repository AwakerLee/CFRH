from JDSH import JDSH
from utils import logger
from args import config

def log_info(logger, config):

    logger.info('--- Configs List---')
    logger.info('--- Dataset:{}'.format(config.DATASET))
    logger.info('--- Train:{}'.format(config.TRAIN))
    logger.info('--- Bit:{}'.format(config.HASH_BIT))
    logger.info('--- Beta:{}'.format(config.beta))
    logger.info('--- Gamma:{}'.format(config.gamma))
    logger.info('--- Eta:{}'.format(config.eta))
    logger.info('--- Delte:{}'.format(config.delte))
    logger.info('--- Phi:{}'.format(config.phi))
    logger.info('--- Lambda:{}'.format(config.lamb))
    logger.info('--- Mu:{}'.format(config.mu))
    logger.info('--- Batch:{}'.format(config.BATCH_SIZE))
    logger.info('--- Topk:{}'.format(config.topk))
    logger.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    logger.info('--- Lr_TXT:{}'.format(config.LR_TXT))


def main():
        etas = [2, 3, 4]
        for eta in etas:
            config.eta = eta
            log = logger()

            log_info(log, config)

            Model = JDSH(log, config)

            if config.TRAIN == False:
                Model.load_checkpoints(config.CHECKPOINT)
                Model.eval()

            else:
                for epoch in range(config.NUM_EPOCH):
                    Model.train(epoch)
                    if (epoch + 1) % config.EVAL_INTERVAL == 0:
                        Model.eval()
                    # save the model
                    if epoch + 1 == config.NUM_EPOCH:
                        Model.save_checkpoints(file_name=config.CHECKPOINT)




if __name__ == '__main__':
    main()
