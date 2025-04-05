/*
 Navicat Premium Dump SQL

 Source Server         : yu
 Source Server Type    : MySQL
 Source Server Version : 80403 (8.4.3)
 Source Host           : localhost:3306
 Source Schema         : captcha

 Target Server Type    : MySQL
 Target Server Version : 80403 (8.4.3)
 File Encoding         : 65001

 Date: 06/04/2025 01:42:28
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for captcha_prediction
-- ----------------------------
DROP TABLE IF EXISTS `captcha_prediction`;
CREATE TABLE `captcha_prediction`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '预测结果ID',
  `pred` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '预测结果',
  `username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户名',
  `model` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型',
  `captcha_type` varchar(155) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '验证码类型',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '0' COMMENT '租户状态（0正常 1停用）',
  `del_flag` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '0' COMMENT '删除标志（0代表存在 2代表删除）',
  `create_dept` bigint NULL DEFAULT NULL COMMENT '创建部门',
  `create_by` bigint NULL DEFAULT NULL COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` bigint NULL DEFAULT NULL COMMENT '更新者',
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `tenant_id` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '000000' COMMENT '租户id',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3341 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '预测记录表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of captcha_prediction
-- ----------------------------
INSERT INTO `captcha_prediction` VALUES (3332, 'TzMg', 'admin', 'CNN', '英文验证码', '0', '0', NULL, NULL, '2025-02-16 23:57:46', NULL, '2025-02-16 23:57:46', '000000');
INSERT INTO `captcha_prediction` VALUES (3333, 'TzMg', 'admin', 'CNN', '英文验证码', '0', '0', NULL, NULL, '2025-02-17 00:48:06', NULL, '2025-02-17 00:48:06', '000000');
INSERT INTO `captcha_prediction` VALUES (3334, 'D:\\code\\python\\Outsourcing\\captcha\\QK_CAPTCHA\\result.png', 'admin', 'YOLOV11', '缺口验证码', '0', '0', NULL, NULL, '2025-02-17 00:48:53', NULL, '2025-02-17 00:48:53', '000000');
INSERT INTO `captcha_prediction` VALUES (3335, 'D:\\code\\python\\Outsourcing\\captcha\\QK_CAPTCHA\\result.png', 'admin', 'YOLOV11', '缺口验证码', '0', '0', NULL, NULL, '2025-02-17 00:48:59', NULL, '2025-02-17 00:48:59', '000000');
INSERT INTO `captcha_prediction` VALUES (3336, 'D:\\code\\python\\Outsourcing\\captcha\\QK_CAPTCHA\\result.png', 'admin', 'YOLOV11', '缺口验证码', '0', '0', NULL, NULL, '2025-02-17 00:54:51', NULL, '2025-02-17 00:54:51', '000000');
INSERT INTO `captcha_prediction` VALUES (3337, 'D:\\Users\\29143\\Desktop\\flask\\front\\src\\assets\\qk_result.png', 'admin', 'YOLOV11', '缺口验证码', '0', '0', NULL, NULL, '2025-02-17 01:10:39', NULL, '2025-02-17 01:10:39', '000000');
INSERT INTO `captcha_prediction` VALUES (3338, 'D:\\Users\\29143\\Desktop\\flask\\front\\src\\assets\\gest_result.png', 'admin', 'UNet', '手势验证码', '0', '0', NULL, NULL, '2025-02-17 01:22:18', NULL, '2025-02-17 01:22:18', '000000');
INSERT INTO `captcha_prediction` VALUES (3339, 'D:\\Users\\29143\\Desktop\\flask\\front\\src\\assets\\gest_result.png', 'admin', 'UNet', '手势验证码', '0', '0', NULL, NULL, '2025-02-17 01:23:11', NULL, '2025-02-17 01:23:11', '000000');
INSERT INTO `captcha_prediction` VALUES (3340, '1953', 'admin', 'CNN-GRU', '英文验证码', '0', '0', NULL, NULL, '2025-02-17 01:29:11', NULL, '2025-02-17 01:29:11', '000000');

-- ----------------------------
-- Table structure for captcha_user
-- ----------------------------
DROP TABLE IF EXISTS `captcha_user`;
CREATE TABLE `captcha_user`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '用户id',
  `username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户名',
  `password` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '密码',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `upate_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 11111115 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '用户表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of captcha_user
-- ----------------------------
INSERT INTO `captcha_user` VALUES (11111111, '123', 'JmtgGbYUyE5Z7MsftP+kVhsSGeAzlQYHSCH8GFDPe5aBIRQge1URP+0bX/GOk0nCXRzJC0KEmWx64laqDkTCzYHpv/lJSXus5tYq1B2L2wlV/zhZCy7eZdOIEmCjfmxbYYgMOL/ydopcDylj2bAk/7H9y1FcPX690mGTshs3YMY5RDUGDWy+/AheKxNLajbDDmhuzPGVWQXeoX7Hbq1WWsoR5NooNXY2cG3nIsbDi2wzyFILTmC7KCdhXYqN5GWuxVIXSjgPwyazaWiMYHp695KewFZAAvk86jn75bt6XGT8mxOhnDhPX1N308w8C6YW9cPJ1ffcEwKBh9l2P+5RqQ==', '2025-02-16 15:14:39', '2025-02-16 15:14:39');
INSERT INTO `captcha_user` VALUES (11111112, 'admin', 'NyWJBcn66aECQXVBDxag1rtB4dZuoHEHJ3jLyMfqyFu2k/+Me4I9+AQur3Azm1Yp1z8ZV9hTypSoqJ0i1wqYDfLldv9qUcIWKgD5gRkyI9jdzr8Zaw6LxtvJwIpueEoFCG6Nap3OLj6QePACI8qp8VHXxrZUPeLZ5eI7YrFK3pjPz9vp5XNnuVrlaqADyedYdYcJSbxJeukyJMoxKZcUHTJFdKvMKfxQwebcMXqqC3Ty79qTBHAQJRoINXZzD6ylBoa/DOy8R3vSKPVPB+pAWsJ7XTNvxM3tjptc6WWmJ34tllTlfSaUvd6Aq6hmX1DZA7jFG5ed+BTndh51l+m0tw==', '2025-02-16 15:40:29', '2025-02-16 15:40:29');
INSERT INTO `captcha_user` VALUES (11111113, '123', 'weOC8a7ab5A5evHr0mIVE6n33cSpRbtHjclxFuDHm4WUVgBaw91kr6RWMaeS9Hh0ozpTcioq5wBkn1jMStGQYL3sgVVx2uHbJvkVvFDYM3I6QXI+4Vn8VmsOhprCFLZIhsrs22SQU2CAYSTrNhw+zbqilYBIS4BML4Bpzqm30HgAJZX7HL+MCfdXrqSs/gRex2xyk003DmKxfz36f+L8ji4xeBwMhWQV1kxqKk2CsWQ2jl3SqVzaLEdh8w4zbbhemva66XhsFtj7yVQ8W9UZ4YxVbW7tGgt6ZCA2PSDX5I7ARTu3xKbx9waH+gbJBMV/R+hEFMDxAVenqaAeCTsd+w==', '2025-03-11 00:31:20', '2025-03-11 00:31:20');
INSERT INTO `captcha_user` VALUES (11111114, '2914341011', 'CSCLwyJ2X+xs3iI1cVS1l11poMWnHr0C8OIZysdtEH2O63kIQ8SYCohRAk/VHX+Z1qSh4rNz3ZPN8kLlTRarGUgrZOQajmqWHFrzgC6MTP0suc0ZWD7rRvnhLPlmAP/8Joi009/W8rrtKsX339J8I/fjEA6BEgwEd8K0Af0KM68IuqwGoSYFrBt6vHV5eoayL1N1wCn0pBA63gGiy4V/yGe81jiHTqaY/2d0R/j+FVIdfP26Mnoz0lIUS/sPaQlWnVKEeZANzEEmBDL3ZoEfWNkZuv27ssACtDi574u99l++hmQveH4wfwH6eR0BqRBYVSLYgLAVzS/tZott4a2kaw==', '2025-03-11 00:32:03', '2025-03-11 00:32:03');

SET FOREIGN_KEY_CHECKS = 1;
