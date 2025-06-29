import argparse
import json
import math
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--unreal_pts",
        type=str,
        required=True,
        help="JSON file containing Unreal coordinates",
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        required=True,
        help="Filename for the resulting .xodr file",
    )
    parser.add_argument(
        "--unreal_origin",
        type=str,
        help="Unreal coordinates for the CARLA origin in the following format: 'x0,y0'",
    )
    return parser.parse_args()


def generate_xodr(pts, filename):
    """Generate a valid xodr file that links N coordinates using a road with several segments"""
    total_length = sum(
        math.sqrt((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2)
        for i in range(len(pts) - 1)
    )
    xodr_content = f"""<?xml version="1.0" standalone="yes"?>
        <OpenDRIVE>
        <header revMajor="1" revMinor="4" name="SingleRoadMultiSegment" version="1.00" date="2024-10-15" north="0" south="0" east="0" west="0">
            <geoReference></geoReference>
        </header>
        <road name="MainRoad" length="{total_length}" id="1" junction="-1">
            <planView>
        """



    elevation_profile = """
    <elevationProfile>
"""

    lanes = """
    <lanes>
    <laneSection s="0.0">
        <left>
        <lane id="1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
        </lane>
        </left>
        <center>
        <lane id="0" type="none" level="false"/>
        </center>
        <right>
        <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
        </lane>
        </right>
    </laneSection>
    </lanes>
"""

    total_length = 0.0
    for i in range(len(pts) - 1):
        x1, y1, z1 = pts[i]
        x2, y2, z2 = pts[i + 1]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        hdg = math.atan2(y2 - y1, x2 - x1)
        slope = (z2 - z1) / length if length != 0 else 0.0

        xodr_content += f"""
    <geometry s="{total_length:.3f}" x="{x1:.3f}" y="{y1:.3f}" hdg="{hdg:.5f}" length="{length:.3f}">
        <line />
    </geometry>
"""
        elevation_profile += f"""
    <elevation s="{total_length:.3f}" a="{z1:.3f}" b="{slope:.5f}" c="0.0" d="0.0"/>
"""
        total_length += length

    # Close the sections for planView, elevationProfile, and lanes
    xodr_content += """
    </planView>
"""
    elevation_profile += """
    </elevationProfile>
"""

    xodr_content += elevation_profile
    xodr_content += lanes
    xodr_content += """
</road>
</OpenDRIVE>
"""

    # Write the content to the specified file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(xodr_content)

    print(f"OpenDRIVE file '{filename}' generated successfully!")


def unreal_to_xodr_coordinates(pt, origin):
    """Convert Unreal coordinates to xodr/CARLA"""
    x1, y1, z1 = pt
    x0, y0 = origin
    x1 = (x1 - x0) / 100
    y1 = -(y1 - y0) / 100
    z1 = z1 / 100
    return (x1, y1, z1)


def main():
    args = parse_args()

    with open(args.unreal_pts, "r", encoding="utf-8") as f:
        unreal_pts = json.load(f)

    if args.unreal_origin:
        origin = tuple(float(c) for c in args.unreal_origin.split(","))
        if len(origin) != 2:
            raise ValueError("'unreal_origin' must be set as 'x0,y0'")
    else:
        origin = (0, 0)

    unreal_pts = [unreal_to_xodr_coordinates(pt, origin) for pt in unreal_pts]

    os.makedirs(os.path.dirname(args.out_fname), exist_ok=True)
    generate_xodr(
        unreal_pts,
        filename=args.out_fname,
    )


if __name__ == "__main__":
    main()
