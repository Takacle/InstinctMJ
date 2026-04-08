#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建v4.3版本URDF文件：
1. 基于v4.1（已做对称化处理）
2. 恢复惯量值（取消v4.2中在惯量中的直接更改）
3. 在joint的dynamic标签中添加转子惯量
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_u4_3():
    """创建v4.3版本URDF文件"""
    
    input_file = "/home/byd/Desktop/urdf/v1.1_Whole_u4.0_0304/urdf/v1.1_Whole_u4.1_0304.urdf"
    output_file = "/home/byd/Desktop/urdf/v1.1_Whole_u4.0_0304/urdf/v1.1_Whole_u4.3_0304.urdf"
    
    # 定义关节配置：joint名称 -> 转子惯量
    joint_configs = {
        # 左腿
        "left_hip_pitch_joint": 0.070017,
        "left_hip_roll_joint": 0.070017,
        "left_hip_yaw_joint": 0.048,
        "left_knee_joint": 0.070017,
        # 右腿
        "right_hip_pitch_joint": 0.070017,
        "right_hip_roll_joint": 0.070017,
        "right_hip_yaw_joint": 0.048,
        "right_knee_joint": 0.070017,
    }
    
    # 解析XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # 更新robot名称
    root.set("name", "v1.1_Whole_u4.3_0304")
    
    # 遍历所有joint，添加dynamic标签
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        
        if joint_name in joint_configs:
            rotor_inertia = joint_configs[joint_name]
            
            # 检查是否已有dynamic标签
            dynamic_elem = joint.find('dynamic')
            if dynamic_elem is None:
                # 创建dynamic标签
                dynamic_elem = ET.SubElement(joint, 'dynamic')
            
            # 添加或更新rotor_inertia属性
            # 使用自定义属性存储转子惯量
            dynamic_elem.set('rotor_inertia', f"{rotor_inertia:.6f}")
            
            print(f"Added rotor inertia to {joint_name}: {rotor_inertia:.6f}")
    
    # 保存文件
    # 使用minidom美化输出
    rough_string = ET.tostring(root, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding='utf-8')
    
    # 移除多余的空白行
    lines = pretty_xml.decode('utf-8').split('\n')
    filtered_lines = [line for line in lines if line.strip()]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered_lines))
    
    print(f"\n文件已保存到: {output_file}")

if __name__ == "__main__":
    create_u4_3()
