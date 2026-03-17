import os
import zipfile
from datetime import datetime

def zip_outputs_without_ckpt(
    outputs_dir: str = "outputs",
    zip_name: str | None = None,
) -> None:
    """
    将 outputs 目录打包为 zip 文件，自动跳过所有 .ckpt 文件。

    :param outputs_dir: 要打包的目录（相对或绝对路径）
    :param zip_name: 生成的 zip 文件名（可选），不传则按时间自动生成
    """
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"目录不存在: {outputs_dir!r}")

    # 默认 zip 文件名：outputs_YYYYMMDD_HHMMSS.zip
    if zip_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"outputs_{ts}.zip"

    # 如果给的是相对路径，就转成绝对路径，保证打包路径正确
    outputs_dir = os.path.abspath(outputs_dir)
    root_dir = os.path.dirname(outputs_dir)

    zip_path = os.path.join(root_dir, zip_name)

    print(f"打包目录: {outputs_dir}")
    print(f"输出文件: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder, subdirs, files in os.walk(outputs_dir):
            # 跳过 outputs/{time_xxx}/ckpt 目录（不继续往下遍历）
            rel_from_outputs = os.path.relpath(folder, outputs_dir)
            if rel_from_outputs != ".":
                parts = rel_from_outputs.split(os.sep)
                # 当前在 outputs/{time_xxx} 这一层时，移除其下的 ckpt 子目录
                if len(parts) == 1 and "ckpt" in subdirs:
                    subdirs.remove("ckpt")

            for filename in files:
                file_path = os.path.join(folder, filename)
                # 计算相对 outputs 的路径，检查是否在 ckpt 目录下
                rel_path = os.path.relpath(file_path, start=outputs_dir)
                if os.sep + "ckpt" + os.sep in os.sep + rel_path:
                    continue

                # 计算在 zip 中的相对路径（以项目根目录为起点）
                arcname = os.path.relpath(file_path, start=root_dir)
                zf.write(file_path, arcname=arcname)

    print("打包完成。")

if __name__ == "__main__":
    # 在项目根目录运行本脚本即可
    zip_outputs_without_ckpt(outputs_dir="outputs")