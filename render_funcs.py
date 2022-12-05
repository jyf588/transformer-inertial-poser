# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

from typing import List
import numpy as np
import pybullet as pb
import bullet_client
from bullet_agent import SimAgent

COLOR_OURS = [1.0, 0.8, 0.0, 1.0]
COLOR_JOINT = [0.6, 0.6, 0.6, 1.0]
COLOR_GT = [0.1, 0.7, 0.1, 0.4]
COLOR_TRANSPOSE = [1.0, 0.75, 0.8, 1.0]


def set_color(pb_c, body_id, base_color=None, specular_color=None):
    if base_color:
        for j in range(-1, pb_c.getNumJoints(body_id)):
            pb_c.changeVisualShape(
                body_id,
                j,
                rgbaColor=base_color)
    if specular_color:
        for j in range(-1, pb_c.getNumJoints(body_id)):
            pb_c.changeVisualShape(
                body_id,
                j,
                specularColor=specular_color)


def update_height_field_pb(pb_c, h_data: List[float], scale: float, terrainShape=-1, terrain=-1):
    if terrainShape == -1:
        # need to disable then enable, weird
        pb_c.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

    # h_data_np = np.random.uniform(-0.5, 0.5, (40, 40))
    # h_data_np = np.array([[0.0]*40]*20 + [[0.5]*40]*20) + np.random.uniform(-0.5, 0.5, (40, 40))
    # print(hex(id(h_data)))
    # print(len(h_data))

    numHeightfieldRows = int(np.sqrt(len(h_data)))
    numHeightfieldColumns = numHeightfieldRows

    terrainShape2 = pb_c.createCollisionShape(
        shapeType=pb.GEOM_HEIGHTFIELD,
        flags=0,
        meshScale=[scale, scale, 1],
        heightfieldTextureScaling=1.0,     # not working, ignore
        heightfieldData=h_data,
        numHeightfieldRows=numHeightfieldRows,
        numHeightfieldColumns=numHeightfieldColumns,
        replaceHeightfieldIndex=terrainShape)

    if terrainShape == -1:
        terrain = pb_c.createMultiBody(0, terrainShape2)

        textureId = pb_c.loadTexture("data/grid2_multi.png")
        pb_c.changeVisualShape(terrain, -1, textureUniqueId=textureId, rgbaColor=[1, 1, 1, 1])
        # pb_c.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])

        pb_c.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    # mean = np.mean(h_data)
    pb_c.resetBasePositionAndOrientation(terrain, [0, 0, 0.0], [0, 0, 0, 1])

    return terrainShape2, terrain


def init_viz(_char_info,
             init_grid_list,
             hmap_scale,
             gui=True,
             compare_gt=True,
             viz_h_map=False,
             color=COLOR_OURS,
             color_gt=COLOR_GT):
    m = pb.GUI if gui else pb.DIRECT
    pb_c = bullet_client.BulletClient(connection_mode=m, options='--opengl3')
    pb_c.resetSimulation()

    pb_c.configureDebugVisualizer(
        flag=pb.COV_ENABLE_RGB_BUFFER_PREVIEW,
        enable=0)
    pb_c.configureDebugVisualizer(
        pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
        enable=0)
    pb_c.configureDebugVisualizer(
        pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        enable=0)

    if viz_h_map:
        h_id, h_b_id = update_height_field_pb(pb_c, init_grid_list, hmap_scale, -1, -1)
    else:
        h_id, h_b_id = None, None

    # pb_c.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
    # numHeightfieldRows = 200
    # numHeightfieldColumns = 200
    # heightfieldData = [0.5] * 200 * 100 + [0.0] * 200 * 100
    # mean = np.mean(heightfieldData)
    # terrainShape = pb_c.createCollisionShape(
    #     shapeType=pb.GEOM_HEIGHTFIELD,
    #     flags=0,
    #     meshScale=[.05, .05, 1],
    #     heightfieldTextureScaling=(numHeightfieldRows-1)/2,
    #     heightfieldData=heightfieldData,
    #     numHeightfieldRows=numHeightfieldRows,
    #     numHeightfieldColumns=numHeightfieldColumns)
    # terrain = pb_c.createMultiBody(0, terrainShape)
    # pb_c.resetBasePositionAndOrientation(terrain, [0, 0, mean], [0, 0, 0, 1])
    # pb_c.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
    # pb_c.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    # numHeightfieldRows = 256
    # numHeightfieldColumns = 256
    # heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
    # for j in range(int(numHeightfieldColumns / 2)):
    #     for i in range(int(numHeightfieldRows / 2)):
    #         height = random.uniform(0, 0.5)
    #         heightfieldData[2 * i + 2 * j * numHeightfieldRows] = -0.4
    #         heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = -0.5
    #         heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = -0.4
    #         heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = -0.5
    #
    # terrainShape = pb_c.createCollisionShape(shapeType=pb.GEOM_HEIGHTFIELD, meshScale=[.05, .05, 1],
    #                                       heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
    #                                       heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows,
    #                                       numHeightfieldColumns=numHeightfieldColumns)
    # terrain = pb_c.createMultiBody(0, terrainShape)
    # # pb_c.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

    # h_id = terrainShape

    pb.resetDebugVisualizerCamera(cameraDistance=3.0,
                                  cameraPitch=-12.50,
                                  cameraYaw=90.0,
                                  cameraTargetPosition=[0, 0, 0.7])

    # # TODO: disable for now for speed...
    pb_c.configureDebugVisualizer(
        flag=pb.COV_ENABLE_SHADOWS | \
             pb.COV_ENABLE_RENDERING | \
             pb.COV_ENABLE_WIREFRAME,
        enable=1,
        shadowMapResolution=2048,
        shadowMapIntensity=0.3,
        shadowMapWorldSize=10,
        rgbBackground=[1, 1, 1],
        lightPosition=(5.0, 5.0, 10.0))

    # # Plane, there is also a 'plane10.urdf' in the data folder
    # plane = \
    #     pb_c.loadURDF(
    #         "data/plane10.urdf",
    #         [0, 0, 0.],
    #         [0, 0, 0, 1.0],
    #         useMaximalCoordinates=True)
    #
    # set_color(pb_c, plane, None, [0.05, 0.05, 0.05])

    '''
    The pybullet does not support using different colors for 
    individual visualization shapes included in the same link
    although there exists an argument in changeVisualShape...
    So, extra 'urdf' file was necessary for joint visualzation only
    '''

    ## Ours (link)

    r1 = SimAgent(name='sim_agent_0',
                            pybullet_client=pb_c,
                            model_file="data/amass.urdf",
                            char_info=_char_info,
                            ref_scale=1.0,
                            self_collision=False,
                            # actuation=spd,
                            kinematic_only=True,
                            verbose=True)

    set_color(pb_c, r1._body_id, color, [0.1, 0.1, 0.1])

    ## Ours (joint)

    # r2 = SimAgent(name='sim_agent_0_joint_only',
    #                        pybullet_client=pb_c,
    #                        model_file="data/amass_joint_only.urdf",
    #                        char_info=_char_info,
    #                        ref_scale=1.0,
    #                        self_collision=False,
    #                        # actuation=spd,
    #                        kinematic_only=True,
    #                        verbose=True)
    #
    # set_color(pb_c, r2._body_id, color_joint, [0.1, 0.1, 0.1])

    if compare_gt:
        r3 = SimAgent(name='sim_agent_1',
                                pybullet_client=pb_c,
                                model_file="data/amass.urdf",
                                char_info=_char_info,
                                ref_scale=1.0,
                                self_collision=False,
                                # actuation=spd,
                                kinematic_only=True,
                                verbose=True)
        set_color(pb_c, r3._body_id, color_gt, [0.1, 0.1, 0.1])
    else:
        r3 = None

    p_vids = []
    for _ in range(10):
        color = [1.0, 0.2, 0.2, 1.0]
        visual_id = pb_c.createVisualShape(pb_c.GEOM_SPHERE,
                                           radius=0.06,
                                           rgbaColor=color,
                                           specularColor=[1, 1, 1])
        bid = pb_c.createMultiBody(0.0,
                                   -1,
                                   visual_id,
                                   [100.0, 100.0, 100.0],
                                   [0.0, 0.0, 0.0, 1.0])
        p_vids.append(bid)
        # self._pb.setCollisionFilterGroupMask(bid, -1, 0, 0)

    # input("press")
    pb_c.removeAllUserDebugItems()
    return pb_c, r1, r3, p_vids, h_id, h_b_id

# import time
# from fairmotion.ops import conversions, quaternion
# import importlib.util
# spec = importlib.util.spec_from_file_location("char_info", 'amass_char_info.py')
# char_info = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(char_info)
#
# pb_c, humanoid,  humanoid_gt, p_vids = \
#     init_viz(
#         char_info,
#         color=COLOR_OURS,
#         color_joint=COLOR_JOINT,
#         color_gt=COLOR_GT)
#
# '''
# Although this is not an ideal and efficient way, but ...
# set_pose should be done for both characters because
# we are maintatining two seperate characters for rendering link and joint.
# '''
#
# Q = conversions.R2Q(conversions.Az2R(1.57))
# Q = quaternion.Q_op(Q, op=['change_order'], xyzw_in=False)
# pb_c.resetBasePositionAndOrientation(humanoid._body_id, [1.5, 0, 1], Q)
# # pb_c.resetBasePositionAndOrientation(humanoid_joint._body_id, [0.025, 0, 1], Q)
#
# if humanoid_gt:
#     Q = conversions.R2Q(conversions.Az2R(1.57))
#     Q = quaternion.Q_op(Q, op=['change_order'], xyzw_in=False)
#     pb_c.resetBasePositionAndOrientation(humanoid_gt._body_id, [-1.0, 0, 1], Q)
#
# gravId = pb.addUserDebugParameter("gravity", -10, 10, -10)
# jointIds = []
# paramIds = []
#
# pb_c.setPhysicsEngineParameter(numSolverIterations=10)
#
# pb.setRealTimeSimulation(1)
# while (1):
#   pb.setGravity(0, 0, pb.readUserDebugParameter(gravId))
#   time.sleep(0.01)
