
#define width 640.
#define height 640.

    

// If true, aspect ratio in the original optimization will be preserved, but extra space outside original FOV might reveal artifact.
// If false, FOV is the same as original optimization, but aspect ratio will not be preserved.
#define preserve_aspect_ratio true

// Smaller factor will zoom in the rendering, larger factor will zoom out
float scale_factor = 1.;



#define X float[](7.614499352622475, 3.1945097710729677, 378.79166058476835, 458.7153516165568, 179.79984895097408, 319.06460167191136, 368.7762221221469, 458.70655850309544, 319.0551824920944, 407.1750003775487, 179.78856454826084, 156.709741730322, 330.63636119019304, 238.58428033850535, 239.37154041756955, 482.43827669815255, 336.5666220603916, 402.2904472745597, 158.4576208320762, 318.5550011830251, 401.55444441453295, 322.1162661038381, -29.901307544911802, 107.80794600845492, 107.80085258635114, 107.80286069882712, -28.34651884827361, 107.81714907786784, 107.81466957968608, -13.937130146182147, 107.79605545226798, -58.39099486614659, -2.058934132495759, -2.6206440738338728, -3.3606938530209938, 2.694846475787397, 2.788319257735719, -2.9361126184934783, 2.997275362187409, -4.899071927460911, -2.72141592316455, -1.1396351628244696)


#define p_0_idx 0
float p_0 = X[p_0_idx];

#define p_1_idx 1
float p_1 = X[p_1_idx];

#define ring_0_pos_0_idx 2
float ring_0_pos_0 = X[ring_0_pos_0_idx];

#define ring_1_pos_0_idx 3
float ring_1_pos_0 = X[ring_1_pos_0_idx];

#define ring_2_pos_0_idx 4
float ring_2_pos_0 = X[ring_2_pos_0_idx];

#define ring_3_pos_0_idx 5
float ring_3_pos_0 = X[ring_3_pos_0_idx];

#define ring_4_pos_0_idx 6
float ring_4_pos_0 = X[ring_4_pos_0_idx];

#define ring_5_pos_0_idx 7
float ring_5_pos_0 = X[ring_5_pos_0_idx];

#define ring_6_pos_0_idx 8
float ring_6_pos_0 = X[ring_6_pos_0_idx];

#define ring_7_pos_0_idx 9
float ring_7_pos_0 = X[ring_7_pos_0_idx];

#define ring_8_pos_0_idx 10
float ring_8_pos_0 = X[ring_8_pos_0_idx];

#define ring_9_pos_0_idx 11
float ring_9_pos_0 = X[ring_9_pos_0_idx];

#define ring_0_pos_1_idx 12
float ring_0_pos_1 = X[ring_0_pos_1_idx];

#define ring_1_pos_1_idx 13
float ring_1_pos_1 = X[ring_1_pos_1_idx];

#define ring_2_pos_1_idx 14
float ring_2_pos_1 = X[ring_2_pos_1_idx];

#define ring_3_pos_1_idx 15
float ring_3_pos_1 = X[ring_3_pos_1_idx];

#define ring_4_pos_1_idx 16
float ring_4_pos_1 = X[ring_4_pos_1_idx];

#define ring_5_pos_1_idx 17
float ring_5_pos_1 = X[ring_5_pos_1_idx];

#define ring_6_pos_1_idx 18
float ring_6_pos_1 = X[ring_6_pos_1_idx];

#define ring_7_pos_1_idx 19
float ring_7_pos_1 = X[ring_7_pos_1_idx];

#define ring_8_pos_1_idx 20
float ring_8_pos_1 = X[ring_8_pos_1_idx];

#define ring_9_pos_1_idx 21
float ring_9_pos_1 = X[ring_9_pos_1_idx];

#define ring_0_radius_idx 22
float ring_0_radius = X[ring_0_radius_idx];

#define ring_1_radius_idx 23
float ring_1_radius = X[ring_1_radius_idx];

#define ring_2_radius_idx 24
float ring_2_radius = X[ring_2_radius_idx];

#define ring_3_radius_idx 25
float ring_3_radius = X[ring_3_radius_idx];

#define ring_4_radius_idx 26
float ring_4_radius = X[ring_4_radius_idx];

#define ring_5_radius_idx 27
float ring_5_radius = X[ring_5_radius_idx];

#define ring_6_radius_idx 28
float ring_6_radius = X[ring_6_radius_idx];

#define ring_7_radius_idx 29
float ring_7_radius = X[ring_7_radius_idx];

#define ring_8_radius_idx 30
float ring_8_radius = X[ring_8_radius_idx];

#define ring_9_radius_idx 31
float ring_9_radius = X[ring_9_radius_idx];

#define ring_0_tilt_idx 32
float ring_0_tilt = X[ring_0_tilt_idx];

#define ring_1_tilt_idx 33
float ring_1_tilt = X[ring_1_tilt_idx];

#define ring_2_tilt_idx 34
float ring_2_tilt = X[ring_2_tilt_idx];

#define ring_3_tilt_idx 35
float ring_3_tilt = X[ring_3_tilt_idx];

#define ring_4_tilt_idx 36
float ring_4_tilt = X[ring_4_tilt_idx];

#define ring_5_tilt_idx 37
float ring_5_tilt = X[ring_5_tilt_idx];

#define ring_6_tilt_idx 38
float ring_6_tilt = X[ring_6_tilt_idx];

#define ring_7_tilt_idx 39
float ring_7_tilt = X[ring_7_tilt_idx];

#define ring_8_tilt_idx 40
float ring_8_tilt = X[ring_8_tilt_idx];

#define ring_9_tilt_idx 41
float ring_9_tilt = X[ring_9_tilt_idx];

void animate_params() {
}
                    

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {

    fragCoord.y = iResolution.y - fragCoord.y;
    
    float current_u;
    float current_v;
    
    if (preserve_aspect_ratio) {
        float max_scale = max(width / iResolution.x, height / iResolution.y) * scale_factor;
        
        vec2 padding = (vec2(width, height) - max_scale * iResolution.xy) / 2.;
        
        current_u = fragCoord.x * max_scale + padding.x;
        current_v = fragCoord.y * max_scale + padding.y;
    } else {
        current_u = fragCoord.x / iResolution.x * width;
        current_v = fragCoord.y / iResolution.y * height;
    }

                animate_params();
    
    float var00014_rel_pos_9_x = ((current_u)-(float(ring_9_pos_0)));
    float var00012 = ((var00014_rel_pos_9_x)*(var00014_rel_pos_9_x));
    float var00015_rel_pos_9_y = ((current_v)-(float(ring_9_pos_1)));
    float var00013 = ((var00015_rel_pos_9_y)*(var00015_rel_pos_9_y));
    float var00011_dist2_ring_9 = ((var00012)+(var00013));
    float var00010_dist_ring_9 = sqrt(var00011_dist2_ring_9);
    float var00009 = ((var00010_dist_ring_9)-(float(ring_9_radius)));
    float var00007_abs = abs(var00009);               
    float var00016 = ((float(p_0))*(float(p_0)));     
    float var00008 = ((var00016)*(float(0.5)));       
    float var00006_cond0_diff_ring_9 = ((var00007_abs)-(var00008));
    bool var00004_cond0_ring_9 = ((var00006_cond0_diff_ring_9)<(float(0)));
    float var00018_phase_raw_ring_9 = ((var00014_rel_pos_9_x)*(float(ring_9_tilt)));
    float var00032_rel_pos_8_x = ((current_u)-(float(ring_8_pos_0)));
    float var00030 = ((var00032_rel_pos_8_x)*(var00032_rel_pos_8_x));
    float var00033_rel_pos_8_y = ((current_v)-(float(ring_8_pos_1)));
    float var00031 = ((var00033_rel_pos_8_y)*(var00033_rel_pos_8_y));
    float var00029_dist2_ring_8 = ((var00030)+(var00031));
    float var00028_dist_ring_8 = sqrt(var00029_dist2_ring_8);
    float var00027 = ((var00028_dist_ring_8)-(float(ring_8_radius)));
    float var00026_abs = abs(var00027);               
    float var00025_cond0_diff_ring_8 = ((var00026_abs)-(var00008));
    bool var00023_cond0_ring_8 = ((var00025_cond0_diff_ring_8)<(float(0)));
    float var00021_phase_raw_ring_8 = ((var00032_rel_pos_8_x)*(float(ring_8_tilt)));
    float var00047_rel_pos_7_x = ((current_u)-(float(ring_7_pos_0)));
    float var00045 = ((var00047_rel_pos_7_x)*(var00047_rel_pos_7_x));
    float var00048_rel_pos_7_y = ((current_v)-(float(ring_7_pos_1)));
    float var00046 = ((var00048_rel_pos_7_y)*(var00048_rel_pos_7_y));
    float var00044_dist2_ring_7 = ((var00045)+(var00046));
    float var00043_dist_ring_7 = sqrt(var00044_dist2_ring_7);
    float var00042 = ((var00043_dist_ring_7)-(float(ring_7_radius)));
    float var00041_abs = abs(var00042);               
    float var00040_cond0_diff_ring_7 = ((var00041_abs)-(var00008));
    bool var00038_cond0_ring_7 = ((var00040_cond0_diff_ring_7)<(float(0)));
    float var00036_phase_raw_ring_7 = ((var00047_rel_pos_7_x)*(float(ring_7_tilt)));
    float var00062_rel_pos_6_x = ((current_u)-(float(ring_6_pos_0)));
    float var00060 = ((var00062_rel_pos_6_x)*(var00062_rel_pos_6_x));
    float var00063_rel_pos_6_y = ((current_v)-(float(ring_6_pos_1)));
    float var00061 = ((var00063_rel_pos_6_y)*(var00063_rel_pos_6_y));
    float var00059_dist2_ring_6 = ((var00060)+(var00061));
    float var00058_dist_ring_6 = sqrt(var00059_dist2_ring_6);
    float var00057 = ((var00058_dist_ring_6)-(float(ring_6_radius)));
    float var00056_abs = abs(var00057);               
    float var00055_cond0_diff_ring_6 = ((var00056_abs)-(var00008));
    bool var00053_cond0_ring_6 = ((var00055_cond0_diff_ring_6)<(float(0)));
    float var00051_phase_raw_ring_6 = ((var00062_rel_pos_6_x)*(float(ring_6_tilt)));
    float var00077_rel_pos_5_x = ((current_u)-(float(ring_5_pos_0)));
    float var00075 = ((var00077_rel_pos_5_x)*(var00077_rel_pos_5_x));
    float var00078_rel_pos_5_y = ((current_v)-(float(ring_5_pos_1)));
    float var00076 = ((var00078_rel_pos_5_y)*(var00078_rel_pos_5_y));
    float var00074_dist2_ring_5 = ((var00075)+(var00076));
    float var00073_dist_ring_5 = sqrt(var00074_dist2_ring_5);
    float var00072 = ((var00073_dist_ring_5)-(float(ring_5_radius)));
    float var00071_abs = abs(var00072);               
    float var00070_cond0_diff_ring_5 = ((var00071_abs)-(var00008));
    bool var00068_cond0_ring_5 = ((var00070_cond0_diff_ring_5)<(float(0)));
    float var00066_phase_raw_ring_5 = ((var00077_rel_pos_5_x)*(float(ring_5_tilt)));
    float var00092_rel_pos_4_x = ((current_u)-(float(ring_4_pos_0)));
    float var00090 = ((var00092_rel_pos_4_x)*(var00092_rel_pos_4_x));
    float var00093_rel_pos_4_y = ((current_v)-(float(ring_4_pos_1)));
    float var00091 = ((var00093_rel_pos_4_y)*(var00093_rel_pos_4_y));
    float var00089_dist2_ring_4 = ((var00090)+(var00091));
    float var00088_dist_ring_4 = sqrt(var00089_dist2_ring_4);
    float var00087 = ((var00088_dist_ring_4)-(float(ring_4_radius)));
    float var00086_abs = abs(var00087);               
    float var00085_cond0_diff_ring_4 = ((var00086_abs)-(var00008));
    bool var00083_cond0_ring_4 = ((var00085_cond0_diff_ring_4)<(float(0)));
    float var00081_phase_raw_ring_4 = ((var00092_rel_pos_4_x)*(float(ring_4_tilt)));
    float var00107_rel_pos_3_x = ((current_u)-(float(ring_3_pos_0)));
    float var00105 = ((var00107_rel_pos_3_x)*(var00107_rel_pos_3_x));
    float var00108_rel_pos_3_y = ((current_v)-(float(ring_3_pos_1)));
    float var00106 = ((var00108_rel_pos_3_y)*(var00108_rel_pos_3_y));
    float var00104_dist2_ring_3 = ((var00105)+(var00106));
    float var00103_dist_ring_3 = sqrt(var00104_dist2_ring_3);
    float var00102 = ((var00103_dist_ring_3)-(float(ring_3_radius)));
    float var00101_abs = abs(var00102);               
    float var00100_cond0_diff_ring_3 = ((var00101_abs)-(var00008));
    bool var00098_cond0_ring_3 = ((var00100_cond0_diff_ring_3)<(float(0)));
    float var00096_phase_raw_ring_3 = ((var00107_rel_pos_3_x)*(float(ring_3_tilt)));
    float var00122_rel_pos_2_x = ((current_u)-(float(ring_2_pos_0)));
    float var00120 = ((var00122_rel_pos_2_x)*(var00122_rel_pos_2_x));
    float var00123_rel_pos_2_y = ((current_v)-(float(ring_2_pos_1)));
    float var00121 = ((var00123_rel_pos_2_y)*(var00123_rel_pos_2_y));
    float var00119_dist2_ring_2 = ((var00120)+(var00121));
    float var00118_dist_ring_2 = sqrt(var00119_dist2_ring_2);
    float var00117 = ((var00118_dist_ring_2)-(float(ring_2_radius)));
    float var00116_abs = abs(var00117);               
    float var00115_cond0_diff_ring_2 = ((var00116_abs)-(var00008));
    bool var00113_cond0_ring_2 = ((var00115_cond0_diff_ring_2)<(float(0)));
    float var00111_phase_raw_ring_2 = ((var00122_rel_pos_2_x)*(float(ring_2_tilt)));
    float var00137_rel_pos_1_x = ((current_u)-(float(ring_1_pos_0)));
    float var00135 = ((var00137_rel_pos_1_x)*(var00137_rel_pos_1_x));
    float var00138_rel_pos_1_y = ((current_v)-(float(ring_1_pos_1)));
    float var00136 = ((var00138_rel_pos_1_y)*(var00138_rel_pos_1_y));
    float var00134_dist2_ring_1 = ((var00135)+(var00136));
    float var00133_dist_ring_1 = sqrt(var00134_dist2_ring_1);
    float var00132 = ((var00133_dist_ring_1)-(float(ring_1_radius)));
    float var00131_abs = abs(var00132);               
    float var00130_cond0_diff_ring_1 = ((var00131_abs)-(var00008));
    bool var00128_cond0_ring_1 = ((var00130_cond0_diff_ring_1)<(float(0)));
    float var00126_phase_raw_ring_1 = ((var00137_rel_pos_1_x)*(float(ring_1_tilt)));
    float var00151_rel_pos_0_x = ((current_u)-(float(ring_0_pos_0)));
    float var00149 = ((var00151_rel_pos_0_x)*(var00151_rel_pos_0_x));
    float var00152_rel_pos_0_y = ((current_v)-(float(ring_0_pos_1)));
    float var00150 = ((var00152_rel_pos_0_y)*(var00152_rel_pos_0_y));
    float var00148_dist2_ring_0 = ((var00149)+(var00150));
    float var00147_dist_ring_0 = sqrt(var00148_dist2_ring_0);
    float var00146 = ((var00147_dist_ring_0)-(float(ring_0_radius)));
    float var00145_abs = abs(var00146);               
    float var00144_cond0_diff_ring_0 = ((var00145_abs)-(var00008));
    bool var00142_cond0_ring_0 = ((var00144_cond0_diff_ring_0)<(float(0)));
    float var00141_phase_raw_ring_0 = ((var00151_rel_pos_0_x)*(float(ring_0_tilt)));
    float var00153_phase_diff_ring_0 = ((var00141_phase_raw_ring_0)-(float(-10000.0)));
    bool var00143_cond2_ring_0 = ((var00153_phase_diff_ring_0)>(float(0)));
    bool var00140_cond_valid_ring_0 = ((var00142_cond0_ring_0)&&(var00143_cond2_ring_0));
    float var00127_phase_ring_0 = bool(var00140_cond_valid_ring_0) ? var00141_phase_raw_ring_0 : float(-10000.0);
    float var00139_phase_diff_ring_1 = ((var00126_phase_raw_ring_1)-(var00127_phase_ring_0));
    bool var00129_cond2_ring_1 = ((var00139_phase_diff_ring_1)>(float(0)));
    bool var00125_cond_valid_ring_1 = ((var00128_cond0_ring_1)&&(var00129_cond2_ring_1));
    float var00112_phase_ring_1 = bool(var00125_cond_valid_ring_1) ? var00126_phase_raw_ring_1 : var00127_phase_ring_0;
    float var00124_phase_diff_ring_2 = ((var00111_phase_raw_ring_2)-(var00112_phase_ring_1));
    bool var00114_cond2_ring_2 = ((var00124_phase_diff_ring_2)>(float(0)));
    bool var00110_cond_valid_ring_2 = ((var00113_cond0_ring_2)&&(var00114_cond2_ring_2));
    float var00097_phase_ring_2 = bool(var00110_cond_valid_ring_2) ? var00111_phase_raw_ring_2 : var00112_phase_ring_1;
    float var00109_phase_diff_ring_3 = ((var00096_phase_raw_ring_3)-(var00097_phase_ring_2));
    bool var00099_cond2_ring_3 = ((var00109_phase_diff_ring_3)>(float(0)));
    bool var00095_cond_valid_ring_3 = ((var00098_cond0_ring_3)&&(var00099_cond2_ring_3));
    float var00082_phase_ring_3 = bool(var00095_cond_valid_ring_3) ? var00096_phase_raw_ring_3 : var00097_phase_ring_2;
    float var00094_phase_diff_ring_4 = ((var00081_phase_raw_ring_4)-(var00082_phase_ring_3));
    bool var00084_cond2_ring_4 = ((var00094_phase_diff_ring_4)>(float(0)));
    bool var00080_cond_valid_ring_4 = ((var00083_cond0_ring_4)&&(var00084_cond2_ring_4));
    float var00067_phase_ring_4 = bool(var00080_cond_valid_ring_4) ? var00081_phase_raw_ring_4 : var00082_phase_ring_3;
    float var00079_phase_diff_ring_5 = ((var00066_phase_raw_ring_5)-(var00067_phase_ring_4));
    bool var00069_cond2_ring_5 = ((var00079_phase_diff_ring_5)>(float(0)));
    bool var00065_cond_valid_ring_5 = ((var00068_cond0_ring_5)&&(var00069_cond2_ring_5));
    float var00052_phase_ring_5 = bool(var00065_cond_valid_ring_5) ? var00066_phase_raw_ring_5 : var00067_phase_ring_4;
    float var00064_phase_diff_ring_6 = ((var00051_phase_raw_ring_6)-(var00052_phase_ring_5));
    bool var00054_cond2_ring_6 = ((var00064_phase_diff_ring_6)>(float(0)));
    bool var00050_cond_valid_ring_6 = ((var00053_cond0_ring_6)&&(var00054_cond2_ring_6));
    float var00037_phase_ring_6 = bool(var00050_cond_valid_ring_6) ? var00051_phase_raw_ring_6 : var00052_phase_ring_5;
    float var00049_phase_diff_ring_7 = ((var00036_phase_raw_ring_7)-(var00037_phase_ring_6));
    bool var00039_cond2_ring_7 = ((var00049_phase_diff_ring_7)>(float(0)));
    bool var00035_cond_valid_ring_7 = ((var00038_cond0_ring_7)&&(var00039_cond2_ring_7));
    float var00022_phase_ring_7 = bool(var00035_cond_valid_ring_7) ? var00036_phase_raw_ring_7 : var00037_phase_ring_6;
    float var00034_phase_diff_ring_8 = ((var00021_phase_raw_ring_8)-(var00022_phase_ring_7));
    bool var00024_cond2_ring_8 = ((var00034_phase_diff_ring_8)>(float(0)));
    bool var00020_cond_valid_ring_8 = ((var00023_cond0_ring_8)&&(var00024_cond2_ring_8));
    float var00019_phase_ring_8 = bool(var00020_cond_valid_ring_8) ? var00021_phase_raw_ring_8 : var00022_phase_ring_7;
    float var00017_phase_diff_ring_9 = ((var00018_phase_raw_ring_9)-(var00019_phase_ring_8));
    bool var00005_cond2_ring_9 = ((var00017_phase_diff_ring_9)>(float(0)));
    bool var00001_cond_valid_ring_9 = ((var00004_cond0_ring_9)&&(var00005_cond2_ring_9));
    float var00158 = ((float(p_1))*(float(p_1)));     
    float var00157_cond1_diff_ring_9 = ((var00006_cond0_diff_ring_9)+(var00158));
    bool var00154_cond1_ring_9 = ((var00157_cond1_diff_ring_9)>(float(0)));
    vec3 var00155 = vec3(float(float(0.0)), float(float(0.0)), float(float(0.0)));
    vec3 var00156_fill_col = vec3(float(float(1.0)), float(float(1.0)), float(float(1.0)));
    vec3 var00002_col_current_ring_9 = bool(var00154_cond1_ring_9) ? var00155 : var00156_fill_col;
    float var00162_cond1_diff_ring_8 = ((var00025_cond0_diff_ring_8)+(var00158));
    bool var00161_cond1_ring_8 = ((var00162_cond1_diff_ring_8)>(float(0)));
    vec3 var00159_col_current_ring_8 = bool(var00161_cond1_ring_8) ? var00155 : var00156_fill_col;
    float var00166_cond1_diff_ring_7 = ((var00040_cond0_diff_ring_7)+(var00158));
    bool var00165_cond1_ring_7 = ((var00166_cond1_diff_ring_7)>(float(0)));
    vec3 var00163_col_current_ring_7 = bool(var00165_cond1_ring_7) ? var00155 : var00156_fill_col;
    float var00170_cond1_diff_ring_6 = ((var00055_cond0_diff_ring_6)+(var00158));
    bool var00169_cond1_ring_6 = ((var00170_cond1_diff_ring_6)>(float(0)));
    vec3 var00167_col_current_ring_6 = bool(var00169_cond1_ring_6) ? var00155 : var00156_fill_col;
    float var00174_cond1_diff_ring_5 = ((var00070_cond0_diff_ring_5)+(var00158));
    bool var00173_cond1_ring_5 = ((var00174_cond1_diff_ring_5)>(float(0)));
    vec3 var00171_col_current_ring_5 = bool(var00173_cond1_ring_5) ? var00155 : var00156_fill_col;
    float var00178_cond1_diff_ring_4 = ((var00085_cond0_diff_ring_4)+(var00158));
    bool var00177_cond1_ring_4 = ((var00178_cond1_diff_ring_4)>(float(0)));
    vec3 var00175_col_current_ring_4 = bool(var00177_cond1_ring_4) ? var00155 : var00156_fill_col;
    float var00182_cond1_diff_ring_3 = ((var00100_cond0_diff_ring_3)+(var00158));
    bool var00181_cond1_ring_3 = ((var00182_cond1_diff_ring_3)>(float(0)));
    vec3 var00179_col_current_ring_3 = bool(var00181_cond1_ring_3) ? var00155 : var00156_fill_col;
    float var00186_cond1_diff_ring_2 = ((var00115_cond0_diff_ring_2)+(var00158));
    bool var00185_cond1_ring_2 = ((var00186_cond1_diff_ring_2)>(float(0)));
    vec3 var00183_col_current_ring_2 = bool(var00185_cond1_ring_2) ? var00155 : var00156_fill_col;
    float var00190_cond1_diff_ring_1 = ((var00130_cond0_diff_ring_1)+(var00158));
    bool var00189_cond1_ring_1 = ((var00190_cond1_diff_ring_1)>(float(0)));
    vec3 var00187_col_current_ring_1 = bool(var00189_cond1_ring_1) ? var00155 : var00156_fill_col;
    float var00194_cond1_diff_ring_0 = ((var00144_cond0_diff_ring_0)+(var00158));
    bool var00193_cond1_ring_0 = ((var00194_cond1_diff_ring_0)>(float(0)));
    vec3 var00191_col_current_ring_0 = bool(var00193_cond1_ring_0) ? var00155 : var00156_fill_col;
    vec3 var00192 = vec3(float(float(1)), float(float(1)), float(float(1)));
    vec3 var00188_col_ring_0 = bool(var00140_cond_valid_ring_0) ? var00191_col_current_ring_0 : var00192;
    vec3 var00184_col_ring_1 = bool(var00125_cond_valid_ring_1) ? var00187_col_current_ring_1 : var00188_col_ring_0;
    vec3 var00180_col_ring_2 = bool(var00110_cond_valid_ring_2) ? var00183_col_current_ring_2 : var00184_col_ring_1;
    vec3 var00176_col_ring_3 = bool(var00095_cond_valid_ring_3) ? var00179_col_current_ring_3 : var00180_col_ring_2;
    vec3 var00172_col_ring_4 = bool(var00080_cond_valid_ring_4) ? var00175_col_current_ring_4 : var00176_col_ring_3;
    vec3 var00168_col_ring_5 = bool(var00065_cond_valid_ring_5) ? var00171_col_current_ring_5 : var00172_col_ring_4;
    vec3 var00164_col_ring_6 = bool(var00050_cond_valid_ring_6) ? var00167_col_current_ring_6 : var00168_col_ring_5;
    vec3 var00160_col_ring_7 = bool(var00035_cond_valid_ring_7) ? var00163_col_current_ring_7 : var00164_col_ring_6;
    vec3 var00003_col_ring_8 = bool(var00020_cond_valid_ring_8) ? var00159_col_current_ring_8 : var00160_col_ring_7;
    vec3 var00000_col_ring_9 = bool(var00001_cond_valid_ring_9) ? var00002_col_current_ring_9 : var00003_col_ring_8;
    
        fragColor = vec4(var00000_col_ring_9, 1.0);
        return;
    }