import pickle

# --- 这里修改成你的 .pkl 文件路径 ---
prediction_path = r'F:\ITD\tools\work_dirs\lw_hrnet_centernet\20250609_171041\test_result\test_result.pkl'
# ------------------------------------

print(f"🔍 正在加载结果文件: {prediction_path}")

try:
    with open(prediction_path, 'rb') as f:
        outputs = pickle.load(f)

    print(f"\n✅ 文件加载成功!")
    print(f"------------------------------------")

    if not outputs:
        print("❌ 错误: 结果文件是空的！里面没有任何数据。")
    else:
        print(f"📊 文件包含 {len(outputs)} 张图片的预测结果。")

        # 检查第一张图片的结果
        first_result = outputs[0]
        print("\n--- 检查第一张图片的信息 ---")

        # 检查图像路径
        if 'img_path' in first_result:
            print(f"🖼️  记录的图像路径是: {first_result['img_path']}")
            print("   (请确认这个路径是否正确，并且图片文件是否存在)")
        else:
             print("⚠️  警告: 在结果中找不到 'img_path' 键。")

        # 检查预测实例
        if 'pred_instances' in first_result:
            pred_scores = first_result['pred_instances']['scores']
            if len(pred_scores) > 0:
                print(f"🎯  第一张图片有 {len(pred_scores)} 个预测目标。")
                print(f"   最高置信度是: {max(pred_scores):.4f}")
                print(f"   最低置信度是: {min(pred_scores):.4f}")
            else:
                print("❌ 错误: 第一张图片没有检测到任何目标 (预测列表为空)。")
        else:
            print("❌ 错误: 在结果中找不到 'pred_instances' 键。")

except Exception as e:
    print(f"\n❌ 加载或分析文件时发生严重错误: {e}")