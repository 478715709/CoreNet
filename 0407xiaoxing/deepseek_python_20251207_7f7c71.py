"""
完整的训练脚本
"""

import argparse
import yaml
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train IM-Fuse model')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to BraTS2023 dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for logs and models')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    train_dataset = BraTS2023Dataset(
        data_dir=args.data_dir,
        split='train',
        modalities=config['modalities'],
        image_size=config['image_size']
    )
    
    val_dataset = BraTS2023Dataset(
        data_dir=args.data_dir,
        split='val',
        modalities=config['modalities'],
        image_size=config['image_size']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建模型
    model = IMFuse(
        num_modalities=len(config['modalities']),
        in_channels=1,
        base_channels=config['base_channels'],
        num_stages=config['num_stages'],
        d_model=config['d_model'],
        num_classes=config['num_classes'],
        use_interleaved=config['use_interleaved']
    ).to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # 损失函数和优化器
    loss_fn = IMFuseLoss(
        class_weights=torch.tensor(config['class_weights']),
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # 训练循环
    best_val_dice = 0
    for epoch in range(config['num_epochs']):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, 
            device, config, epoch
        )
        
        # 验证
        val_dice = validate(
            model, val_loader, device, config
        )
        
        # 更新学习率
        scheduler.step(val_dice)
        
        # 保存检查点
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(
                model, optimizer, epoch, val_dice,
                output_dir / 'best_model.pth'
            )
        
        # 保存最新检查点
        save_checkpoint(
            model, optimizer, epoch, val_dice,
            output_dir / 'latest_model.pth'
        )
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Dice = {val_dice:.4f}")

if __name__ == "__main__":
    main()