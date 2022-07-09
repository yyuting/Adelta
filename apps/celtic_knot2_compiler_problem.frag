
#define width 640.
#define height 640.

    

// If true, aspect ratio in the original optimization will be preserved, but extra space outside original FOV might reveal artifact.
// If false, FOV is the same as original optimization, but aspect ratio will not be preserved.
#define preserve_aspect_ratio true

// Smaller factor will zoom in the rendering, larger factor will zoom out
float scale_factor = 1.;



#define X float[](7.614499352622475, 3.1945097710729677, 458.7153516165568, 179.79984895097408, 319.06460167191136, 458.70655850309544, 319.0551824920944, 179.78856454826084, 238.58428033850535, 239.37154041756955, 482.43827669815255, 402.2904472745597, 158.4576208320762, 401.55444441453295, 107.80794600845492, 107.80085258635114, 107.80286069882712, 107.81714907786784, 107.81466957968608, 107.79605545226798, -2.6206440738338728, -3.3606938530209938, 2.694846475787397, -2.9361126184934783, 2.997275362187409, -2.72141592316455)


#define p_0_idx 0
float p_0 = X[p_0_idx];

#define p_1_idx 1
float p_1 = X[p_1_idx];

#define ring_1_pos_0_idx 2
float ring_1_pos_0 = X[ring_1_pos_0_idx];

#define ring_2_pos_0_idx 3
float ring_2_pos_0 = X[ring_2_pos_0_idx];

#define ring_3_pos_0_idx 4
float ring_3_pos_0 = X[ring_3_pos_0_idx];

#define ring_5_pos_0_idx 5
float ring_5_pos_0 = X[ring_5_pos_0_idx];

#define ring_6_pos_0_idx 6
float ring_6_pos_0 = X[ring_6_pos_0_idx];

#define ring_8_pos_0_idx 7
float ring_8_pos_0 = X[ring_8_pos_0_idx];

#define ring_1_pos_1_idx 8
float ring_1_pos_1 = X[ring_1_pos_1_idx];

#define ring_2_pos_1_idx 9
float ring_2_pos_1 = X[ring_2_pos_1_idx];

#define ring_3_pos_1_idx 10
float ring_3_pos_1 = X[ring_3_pos_1_idx];

#define ring_5_pos_1_idx 11
float ring_5_pos_1 = X[ring_5_pos_1_idx];

#define ring_6_pos_1_idx 12
float ring_6_pos_1 = X[ring_6_pos_1_idx];

#define ring_8_pos_1_idx 13
float ring_8_pos_1 = X[ring_8_pos_1_idx];

#define ring_1_radius_idx 14
float ring_1_radius = X[ring_1_radius_idx];

#define ring_2_radius_idx 15
float ring_2_radius = X[ring_2_radius_idx];

#define ring_3_radius_idx 16
float ring_3_radius = X[ring_3_radius_idx];

#define ring_5_radius_idx 17
float ring_5_radius = X[ring_5_radius_idx];

#define ring_6_radius_idx 18
float ring_6_radius = X[ring_6_radius_idx];

#define ring_8_radius_idx 19
float ring_8_radius = X[ring_8_radius_idx];

#define ring_1_tilt_idx 20
float ring_1_tilt = X[ring_1_tilt_idx];

#define ring_2_tilt_idx 21
float ring_2_tilt = X[ring_2_tilt_idx];

#define ring_3_tilt_idx 22
float ring_3_tilt = X[ring_3_tilt_idx];

#define ring_5_tilt_idx 23
float ring_5_tilt = X[ring_5_tilt_idx];

#define ring_6_tilt_idx 24
float ring_6_tilt = X[ring_6_tilt_idx];

#define ring_8_tilt_idx 25
float ring_8_tilt = X[ring_8_tilt_idx];

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
    
    float var00016_rel_pos_8_x = ((current_u)-(float(ring_8_pos_0)));
    float var00014 = ((var00016_rel_pos_8_x)*(var00016_rel_pos_8_x));
    float var00017_rel_pos_8_y = ((current_v)-(float(ring_8_pos_1)));
    float var00015 = ((var00017_rel_pos_8_y)*(var00017_rel_pos_8_y));
    float var00013_dist2_ring_8 = ((var00014)+(var00015));
    float var00012_dist_ring_8 = sqrt(var00013_dist2_ring_8);
    float var00011 = ((var00012_dist_ring_8)-(float(ring_8_radius)));
    float var00009_abs = abs(var00011);               
    float var00018 = ((float(p_0))*(float(p_0)));     
    float var00010 = ((var00018)*(float(0.5)));       
    float var00008_cond0_diff_ring_8 = ((var00009_abs)-(var00010));
    bool var00006_cond0_ring_8 = ((var00008_cond0_diff_ring_8)<(float(0)));
    float var00020_phase_raw_ring_8 = ((var00016_rel_pos_8_x)*(float(ring_8_tilt)));
    float var00036_rel_pos_6_x = ((current_u)-(float(ring_6_pos_0)));
    float var00034 = ((var00036_rel_pos_6_x)*(var00036_rel_pos_6_x));
    float var00037_rel_pos_6_y = ((current_v)-(float(ring_6_pos_1)));
    float var00035 = ((var00037_rel_pos_6_y)*(var00037_rel_pos_6_y));
    float var00033_dist2_ring_6 = ((var00034)+(var00035));
    float var00032_dist_ring_6 = sqrt(var00033_dist2_ring_6);
    float var00031 = ((var00032_dist_ring_6)-(float(ring_6_radius)));
    float var00030_abs = abs(var00031);               
    float var00029_cond0_diff_ring_6 = ((var00030_abs)-(var00010));
    bool var00027_cond0_ring_6 = ((var00029_cond0_diff_ring_6)<(float(0)));
    float var00025_phase_raw_ring_6 = ((var00036_rel_pos_6_x)*(float(ring_6_tilt)));
    float var00052_rel_pos_5_x = ((current_u)-(float(ring_5_pos_0)));
    float var00050 = ((var00052_rel_pos_5_x)*(var00052_rel_pos_5_x));
    float var00053_rel_pos_5_y = ((current_v)-(float(ring_5_pos_1)));
    float var00051 = ((var00053_rel_pos_5_y)*(var00053_rel_pos_5_y));
    float var00049_dist2_ring_5 = ((var00050)+(var00051));
    float var00048_dist_ring_5 = sqrt(var00049_dist2_ring_5);
    float var00047 = ((var00048_dist_ring_5)-(float(ring_5_radius)));
    float var00046_abs = abs(var00047);               
    float var00045_cond0_diff_ring_5 = ((var00046_abs)-(var00010));
    bool var00043_cond0_ring_5 = ((var00045_cond0_diff_ring_5)<(float(0)));
    float var00041_phase_raw_ring_5 = ((var00052_rel_pos_5_x)*(float(ring_5_tilt)));
    float var00069_rel_pos_3_x = ((current_u)-(float(ring_3_pos_0)));
    float var00067 = ((var00069_rel_pos_3_x)*(var00069_rel_pos_3_x));
    float var00070_rel_pos_3_y = ((current_v)-(float(ring_3_pos_1)));
    float var00068 = ((var00070_rel_pos_3_y)*(var00070_rel_pos_3_y));
    float var00066_dist2_ring_3 = ((var00067)+(var00068));
    float var00065_dist_ring_3 = sqrt(var00066_dist2_ring_3);
    float var00064 = ((var00065_dist_ring_3)-(float(ring_3_radius)));
    float var00063_abs = abs(var00064);               
    float var00062_cond0_diff_ring_3 = ((var00063_abs)-(var00010));
    bool var00060_cond0_ring_3 = ((var00062_cond0_diff_ring_3)<(float(0)));
    float var00058_phase_raw_ring_3 = ((var00069_rel_pos_3_x)*(float(ring_3_tilt)));
    float var00085_rel_pos_2_x = ((current_u)-(float(ring_2_pos_0)));
    float var00083 = ((var00085_rel_pos_2_x)*(var00085_rel_pos_2_x));
    float var00086_rel_pos_2_y = ((current_v)-(float(ring_2_pos_1)));
    float var00084 = ((var00086_rel_pos_2_y)*(var00086_rel_pos_2_y));
    float var00082_dist2_ring_2 = ((var00083)+(var00084));
    float var00081_dist_ring_2 = sqrt(var00082_dist2_ring_2);
    float var00080 = ((var00081_dist_ring_2)-(float(ring_2_radius)));
    float var00079_abs = abs(var00080);               
    float var00078_cond0_diff_ring_2 = ((var00079_abs)-(var00010));
    bool var00076_cond0_ring_2 = ((var00078_cond0_diff_ring_2)<(float(0)));
    float var00074_phase_raw_ring_2 = ((var00085_rel_pos_2_x)*(float(ring_2_tilt)));
    float var00101_rel_pos_1_x = ((current_u)-(float(ring_1_pos_0)));
    float var00099 = ((var00101_rel_pos_1_x)*(var00101_rel_pos_1_x));
    float var00102_rel_pos_1_y = ((current_v)-(float(ring_1_pos_1)));
    float var00100 = ((var00102_rel_pos_1_y)*(var00102_rel_pos_1_y));
    float var00098_dist2_ring_1 = ((var00099)+(var00100));
    float var00097_dist_ring_1 = sqrt(var00098_dist2_ring_1);
    float var00096 = ((var00097_dist_ring_1)-(float(ring_1_radius)));
    float var00095_abs = abs(var00096);               
    float var00094_cond0_diff_ring_1 = ((var00095_abs)-(var00010));
    bool var00092_cond0_ring_1 = ((var00094_cond0_diff_ring_1)<(float(0)));
    float var00090_phase_raw_ring_1 = ((var00101_rel_pos_1_x)*(float(ring_1_tilt)));
    float var00091 = float(-10000.0);
    float var00103_phase_diff_ring_1 = ((var00090_phase_raw_ring_1)-(var00091));
    bool var00093_cond2_ring_1 = ((var00103_phase_diff_ring_1)>(float(0)));
    bool var00089_cond_valid_ring_1 = ((var00092_cond0_ring_1)&&(var00093_cond2_ring_1));
    float var00088_phase_ring_1 = bool(var00089_cond_valid_ring_1) ? var00090_phase_raw_ring_1 : var00091;
    float var00075 = var00088_phase_ring_1;
    float var00087_phase_diff_ring_2 = ((var00074_phase_raw_ring_2)-(var00075));
    bool var00077_cond2_ring_2 = ((var00087_phase_diff_ring_2)>(float(0)));
    bool var00073_cond_valid_ring_2 = ((var00076_cond0_ring_2)&&(var00077_cond2_ring_2));
    float var00072_phase_ring_2 = bool(var00073_cond_valid_ring_2) ? var00074_phase_raw_ring_2 : var00075;
    float var00059 = var00072_phase_ring_2;
    float var00071_phase_diff_ring_3 = ((var00058_phase_raw_ring_3)-(var00059));
    bool var00061_cond2_ring_3 = ((var00071_phase_diff_ring_3)>(float(0)));
    bool var00057_cond_valid_ring_3 = ((var00060_cond0_ring_3)&&(var00061_cond2_ring_3));
    float var00056_phase_ring_3 = bool(var00057_cond_valid_ring_3) ? var00058_phase_raw_ring_3 : var00059;
    float var00055 = var00056_phase_ring_3;
    float var00042 = var00055;
    float var00054_phase_diff_ring_5 = ((var00041_phase_raw_ring_5)-(var00042));
    bool var00044_cond2_ring_5 = ((var00054_phase_diff_ring_5)>(float(0)));
    bool var00040_cond_valid_ring_5 = ((var00043_cond0_ring_5)&&(var00044_cond2_ring_5));
    float var00039_phase_ring_5 = bool(var00040_cond_valid_ring_5) ? var00041_phase_raw_ring_5 : var00042;
    float var00026 = var00039_phase_ring_5;
    float var00038_phase_diff_ring_6 = ((var00025_phase_raw_ring_6)-(var00026));
    bool var00028_cond2_ring_6 = ((var00038_phase_diff_ring_6)>(float(0)));
    bool var00024_cond_valid_ring_6 = ((var00027_cond0_ring_6)&&(var00028_cond2_ring_6));
    float var00023_phase_ring_6 = bool(var00024_cond_valid_ring_6) ? var00025_phase_raw_ring_6 : var00026;
    float var00022 = var00023_phase_ring_6;
    float var00021 = var00022;
    float var00019_phase_diff_ring_8 = ((var00020_phase_raw_ring_8)-(var00021));
    bool var00007_cond2_ring_8 = ((var00019_phase_diff_ring_8)>(float(0)));
    bool var00003_cond_valid_ring_8 = ((var00006_cond0_ring_8)&&(var00007_cond2_ring_8));
    float var00108 = ((float(p_1))*(float(p_1)));     
    float var00107_cond1_diff_ring_8 = ((var00008_cond0_diff_ring_8)+(var00108));
    bool var00104_cond1_ring_8 = ((var00107_cond1_diff_ring_8)>(float(0)));
    vec3 var00105 = vec3(float(float(0.0)), float(float(0.0)), float(float(0.0)));
    vec3 var00106_fill_col = vec3(float(float(1.0)), float(float(1.0)), float(float(1.0)));
    vec3 var00004_col_current_ring_8 = bool(var00104_cond1_ring_8) ? var00105 : var00106_fill_col;
    float var00114_cond1_diff_ring_6 = ((var00029_cond0_diff_ring_6)+(var00108));
    bool var00113_cond1_ring_6 = ((var00114_cond1_diff_ring_6)>(float(0)));
    vec3 var00111_col_current_ring_6 = bool(var00113_cond1_ring_6) ? var00105 : var00106_fill_col;
    float var00119_cond1_diff_ring_5 = ((var00045_cond0_diff_ring_5)+(var00108));
    bool var00118_cond1_ring_5 = ((var00119_cond1_diff_ring_5)>(float(0)));
    vec3 var00116_col_current_ring_5 = bool(var00118_cond1_ring_5) ? var00105 : var00106_fill_col;
    float var00125_cond1_diff_ring_3 = ((var00062_cond0_diff_ring_3)+(var00108));
    bool var00124_cond1_ring_3 = ((var00125_cond1_diff_ring_3)>(float(0)));
    vec3 var00122_col_current_ring_3 = bool(var00124_cond1_ring_3) ? var00105 : var00106_fill_col;
    float var00130_cond1_diff_ring_2 = ((var00078_cond0_diff_ring_2)+(var00108));
    bool var00129_cond1_ring_2 = ((var00130_cond1_diff_ring_2)>(float(0)));
    vec3 var00127_col_current_ring_2 = bool(var00129_cond1_ring_2) ? var00105 : var00106_fill_col;
    float var00135_cond1_diff_ring_1 = ((var00094_cond0_diff_ring_1)+(var00108));
    bool var00134_cond1_ring_1 = ((var00135_cond1_diff_ring_1)>(float(0)));
    vec3 var00132_col_current_ring_1 = bool(var00134_cond1_ring_1) ? var00105 : var00106_fill_col;
    vec3 var00136 = vec3(float(float(1)), float(float(1)), float(float(1)));
    vec3 var00133 = var00136;
    vec3 var00131_col_ring_1 = bool(var00089_cond_valid_ring_1) ? var00132_col_current_ring_1 : var00133;
    vec3 var00128 = var00131_col_ring_1;
    vec3 var00126_col_ring_2 = bool(var00073_cond_valid_ring_2) ? var00127_col_current_ring_2 : var00128;
    vec3 var00123 = var00126_col_ring_2;
    vec3 var00121_col_ring_3 = bool(var00057_cond_valid_ring_3) ? var00122_col_current_ring_3 : var00123;
    vec3 var00120 = var00121_col_ring_3;
    vec3 var00117 = var00120;
    vec3 var00115_col_ring_5 = bool(var00040_cond_valid_ring_5) ? var00116_col_current_ring_5 : var00117;
    vec3 var00112 = var00115_col_ring_5;
    vec3 var00110_col_ring_6 = bool(var00024_cond_valid_ring_6) ? var00111_col_current_ring_6 : var00112;
    vec3 var00109 = var00110_col_ring_6;
    vec3 var00005 = var00109;
    vec3 var00002_col_ring_8 = bool(var00003_cond_valid_ring_8) ? var00004_col_current_ring_8 : var00005;
    vec3 var00001 = var00002_col_ring_8;
    vec3 var00000 = var00001;
    
        fragColor = vec4(var00000, 1.0);
        return;
    }