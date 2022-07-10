
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
                    
void animate_uv(
inout float  u, 
inout float  v){}

void animate_ring_col_8(
inout vec3  fill_col, 
in float  rel_pos_8_x, 
in float  rel_pos_8_y){}

void animate_ring_col_6(
inout vec3  fill_col, 
in float  rel_pos_6_x, 
in float  rel_pos_6_y){}

void animate_ring_col_5(
inout vec3  fill_col, 
in float  rel_pos_5_x, 
in float  rel_pos_5_y){}

void animate_ring_col_3(
inout vec3  fill_col, 
in float  rel_pos_3_x, 
in float  rel_pos_3_y){}

void animate_ring_col_2(
inout vec3  fill_col, 
in float  rel_pos_2_x, 
in float  rel_pos_2_y){}

void animate_ring_col_1(
inout vec3  fill_col, 
in float  rel_pos_1_x, 
in float  rel_pos_1_y){}


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
    
    float  var00017_u = current_u;
    float  var00018_v = current_v;
    animate_uv(var00017_u, var00018_v);
    float var00016_rel_pos_8_x = ((var00017_u)-(float(ring_8_pos_0)));
    float var00014 = ((var00016_rel_pos_8_x)*(var00016_rel_pos_8_x));
    float var00019_rel_pos_8_y = ((var00018_v)-(float(ring_8_pos_1)));
    float var00015 = ((var00019_rel_pos_8_y)*(var00019_rel_pos_8_y));
    float var00013_dist2_ring_8 = ((var00014)+(var00015));
    float var00012_dist_ring_8 = sqrt(var00013_dist2_ring_8);
    float var00011 = ((var00012_dist_ring_8)-(float(ring_8_radius)));
    float var00009_abs = abs(var00011);               
    float var00020 = ((float(p_0))*(float(p_0)));     
    float var00010 = ((var00020)*(float(0.5)));       
    float var00008_cond0_diff_ring_8 = ((var00009_abs)-(var00010));
    bool var00006_cond0_ring_8 = ((var00008_cond0_diff_ring_8)<(float(0)));
    float var00022_phase_raw_ring_8 = ((var00016_rel_pos_8_x)*(float(ring_8_tilt)));
    float var00038_rel_pos_6_x = ((var00017_u)-(float(ring_6_pos_0)));
    float var00036 = ((var00038_rel_pos_6_x)*(var00038_rel_pos_6_x));
    float var00039_rel_pos_6_y = ((var00018_v)-(float(ring_6_pos_1)));
    float var00037 = ((var00039_rel_pos_6_y)*(var00039_rel_pos_6_y));
    float var00035_dist2_ring_6 = ((var00036)+(var00037));
    float var00034_dist_ring_6 = sqrt(var00035_dist2_ring_6);
    float var00033 = ((var00034_dist_ring_6)-(float(ring_6_radius)));
    float var00032_abs = abs(var00033);               
    float var00031_cond0_diff_ring_6 = ((var00032_abs)-(var00010));
    bool var00029_cond0_ring_6 = ((var00031_cond0_diff_ring_6)<(float(0)));
    float var00027_phase_raw_ring_6 = ((var00038_rel_pos_6_x)*(float(ring_6_tilt)));
    float var00054_rel_pos_5_x = ((var00017_u)-(float(ring_5_pos_0)));
    float var00052 = ((var00054_rel_pos_5_x)*(var00054_rel_pos_5_x));
    float var00055_rel_pos_5_y = ((var00018_v)-(float(ring_5_pos_1)));
    float var00053 = ((var00055_rel_pos_5_y)*(var00055_rel_pos_5_y));
    float var00051_dist2_ring_5 = ((var00052)+(var00053));
    float var00050_dist_ring_5 = sqrt(var00051_dist2_ring_5);
    float var00049 = ((var00050_dist_ring_5)-(float(ring_5_radius)));
    float var00048_abs = abs(var00049);               
    float var00047_cond0_diff_ring_5 = ((var00048_abs)-(var00010));
    bool var00045_cond0_ring_5 = ((var00047_cond0_diff_ring_5)<(float(0)));
    float var00043_phase_raw_ring_5 = ((var00054_rel_pos_5_x)*(float(ring_5_tilt)));
    float var00071_rel_pos_3_x = ((var00017_u)-(float(ring_3_pos_0)));
    float var00069 = ((var00071_rel_pos_3_x)*(var00071_rel_pos_3_x));
    float var00072_rel_pos_3_y = ((var00018_v)-(float(ring_3_pos_1)));
    float var00070 = ((var00072_rel_pos_3_y)*(var00072_rel_pos_3_y));
    float var00068_dist2_ring_3 = ((var00069)+(var00070));
    float var00067_dist_ring_3 = sqrt(var00068_dist2_ring_3);
    float var00066 = ((var00067_dist_ring_3)-(float(ring_3_radius)));
    float var00065_abs = abs(var00066);               
    float var00064_cond0_diff_ring_3 = ((var00065_abs)-(var00010));
    bool var00062_cond0_ring_3 = ((var00064_cond0_diff_ring_3)<(float(0)));
    float var00060_phase_raw_ring_3 = ((var00071_rel_pos_3_x)*(float(ring_3_tilt)));
    float var00087_rel_pos_2_x = ((var00017_u)-(float(ring_2_pos_0)));
    float var00085 = ((var00087_rel_pos_2_x)*(var00087_rel_pos_2_x));
    float var00088_rel_pos_2_y = ((var00018_v)-(float(ring_2_pos_1)));
    float var00086 = ((var00088_rel_pos_2_y)*(var00088_rel_pos_2_y));
    float var00084_dist2_ring_2 = ((var00085)+(var00086));
    float var00083_dist_ring_2 = sqrt(var00084_dist2_ring_2);
    float var00082 = ((var00083_dist_ring_2)-(float(ring_2_radius)));
    float var00081_abs = abs(var00082);               
    float var00080_cond0_diff_ring_2 = ((var00081_abs)-(var00010));
    bool var00078_cond0_ring_2 = ((var00080_cond0_diff_ring_2)<(float(0)));
    float var00076_phase_raw_ring_2 = ((var00087_rel_pos_2_x)*(float(ring_2_tilt)));
    float var00103_rel_pos_1_x = ((var00017_u)-(float(ring_1_pos_0)));
    float var00101 = ((var00103_rel_pos_1_x)*(var00103_rel_pos_1_x));
    float var00104_rel_pos_1_y = ((var00018_v)-(float(ring_1_pos_1)));
    float var00102 = ((var00104_rel_pos_1_y)*(var00104_rel_pos_1_y));
    float var00100_dist2_ring_1 = ((var00101)+(var00102));
    float var00099_dist_ring_1 = sqrt(var00100_dist2_ring_1);
    float var00098 = ((var00099_dist_ring_1)-(float(ring_1_radius)));
    float var00097_abs = abs(var00098);               
    float var00096_cond0_diff_ring_1 = ((var00097_abs)-(var00010));
    bool var00094_cond0_ring_1 = ((var00096_cond0_diff_ring_1)<(float(0)));
    float var00092_phase_raw_ring_1 = ((var00103_rel_pos_1_x)*(float(ring_1_tilt)));
    float var00093 = float(-10000.0);
    float var00105_phase_diff_ring_1 = ((var00092_phase_raw_ring_1)-(var00093));
    bool var00095_cond2_ring_1 = ((var00105_phase_diff_ring_1)>(float(0)));
    bool var00091_cond_valid_ring_1 = ((var00094_cond0_ring_1)&&(var00095_cond2_ring_1));
    float var00090_phase_ring_1 = bool(var00091_cond_valid_ring_1) ? var00092_phase_raw_ring_1 : var00093;
    float var00077 = var00090_phase_ring_1;
    float var00089_phase_diff_ring_2 = ((var00076_phase_raw_ring_2)-(var00077));
    bool var00079_cond2_ring_2 = ((var00089_phase_diff_ring_2)>(float(0)));
    bool var00075_cond_valid_ring_2 = ((var00078_cond0_ring_2)&&(var00079_cond2_ring_2));
    float var00074_phase_ring_2 = bool(var00075_cond_valid_ring_2) ? var00076_phase_raw_ring_2 : var00077;
    float var00061 = var00074_phase_ring_2;
    float var00073_phase_diff_ring_3 = ((var00060_phase_raw_ring_3)-(var00061));
    bool var00063_cond2_ring_3 = ((var00073_phase_diff_ring_3)>(float(0)));
    bool var00059_cond_valid_ring_3 = ((var00062_cond0_ring_3)&&(var00063_cond2_ring_3));
    float var00058_phase_ring_3 = bool(var00059_cond_valid_ring_3) ? var00060_phase_raw_ring_3 : var00061;
    float var00057 = var00058_phase_ring_3;
    float var00044 = var00057;
    float var00056_phase_diff_ring_5 = ((var00043_phase_raw_ring_5)-(var00044));
    bool var00046_cond2_ring_5 = ((var00056_phase_diff_ring_5)>(float(0)));
    bool var00042_cond_valid_ring_5 = ((var00045_cond0_ring_5)&&(var00046_cond2_ring_5));
    float var00041_phase_ring_5 = bool(var00042_cond_valid_ring_5) ? var00043_phase_raw_ring_5 : var00044;
    float var00028 = var00041_phase_ring_5;
    float var00040_phase_diff_ring_6 = ((var00027_phase_raw_ring_6)-(var00028));
    bool var00030_cond2_ring_6 = ((var00040_phase_diff_ring_6)>(float(0)));
    bool var00026_cond_valid_ring_6 = ((var00029_cond0_ring_6)&&(var00030_cond2_ring_6));
    float var00025_phase_ring_6 = bool(var00026_cond_valid_ring_6) ? var00027_phase_raw_ring_6 : var00028;
    float var00024 = var00025_phase_ring_6;
    float var00023 = var00024;
    float var00021_phase_diff_ring_8 = ((var00022_phase_raw_ring_8)-(var00023));
    bool var00007_cond2_ring_8 = ((var00021_phase_diff_ring_8)>(float(0)));
    bool var00003_cond_valid_ring_8 = ((var00006_cond0_ring_8)&&(var00007_cond2_ring_8));
    float var00110 = ((float(p_1))*(float(p_1)));     
    float var00109_cond1_diff_ring_8 = ((var00008_cond0_diff_ring_8)+(var00110));
    bool var00106_cond1_ring_8 = ((var00109_cond1_diff_ring_8)>(float(0)));
    vec3 var00107 = vec3(float(float(0.0)), float(float(0.0)), float(float(0.0)));
    vec3 var00111_fill_col = vec3(float(float(1.0)), float(float(1.0)), float(float(1.0)));
    vec3  var00108_fill_col = var00111_fill_col;
    animate_ring_col_8(var00108_fill_col, var00016_rel_pos_8_x, var00019_rel_pos_8_y);
    vec3 var00004_col_current_ring_8 = bool(var00106_cond1_ring_8) ? var00107 : var00108_fill_col;
    float var00118_cond1_diff_ring_6 = ((var00031_cond0_diff_ring_6)+(var00110));
    bool var00116_cond1_ring_6 = ((var00118_cond1_diff_ring_6)>(float(0)));
    vec3  var00117_fill_col = var00111_fill_col;
    animate_ring_col_6(var00117_fill_col, var00038_rel_pos_6_x, var00039_rel_pos_6_y);
    vec3 var00114_col_current_ring_6 = bool(var00116_cond1_ring_6) ? var00107 : var00117_fill_col;
    float var00124_cond1_diff_ring_5 = ((var00047_cond0_diff_ring_5)+(var00110));
    bool var00122_cond1_ring_5 = ((var00124_cond1_diff_ring_5)>(float(0)));
    vec3  var00123_fill_col = var00111_fill_col;
    animate_ring_col_5(var00123_fill_col, var00054_rel_pos_5_x, var00055_rel_pos_5_y);
    vec3 var00120_col_current_ring_5 = bool(var00122_cond1_ring_5) ? var00107 : var00123_fill_col;
    float var00131_cond1_diff_ring_3 = ((var00064_cond0_diff_ring_3)+(var00110));
    bool var00129_cond1_ring_3 = ((var00131_cond1_diff_ring_3)>(float(0)));
    vec3  var00130_fill_col = var00111_fill_col;
    animate_ring_col_3(var00130_fill_col, var00071_rel_pos_3_x, var00072_rel_pos_3_y);
    vec3 var00127_col_current_ring_3 = bool(var00129_cond1_ring_3) ? var00107 : var00130_fill_col;
    float var00137_cond1_diff_ring_2 = ((var00080_cond0_diff_ring_2)+(var00110));
    bool var00135_cond1_ring_2 = ((var00137_cond1_diff_ring_2)>(float(0)));
    vec3  var00136_fill_col = var00111_fill_col;
    animate_ring_col_2(var00136_fill_col, var00087_rel_pos_2_x, var00088_rel_pos_2_y);
    vec3 var00133_col_current_ring_2 = bool(var00135_cond1_ring_2) ? var00107 : var00136_fill_col;
    float var00143_cond1_diff_ring_1 = ((var00096_cond0_diff_ring_1)+(var00110));
    bool var00141_cond1_ring_1 = ((var00143_cond1_diff_ring_1)>(float(0)));
    vec3  var00142_fill_col = var00111_fill_col;
    animate_ring_col_1(var00142_fill_col, var00103_rel_pos_1_x, var00104_rel_pos_1_y);
    vec3 var00139_col_current_ring_1 = bool(var00141_cond1_ring_1) ? var00107 : var00142_fill_col;
    vec3 var00144 = vec3(float(float(1)), float(float(1)), float(float(1)));
    vec3 var00140 = var00144;
    vec3 var00138_col_ring_1 = bool(var00091_cond_valid_ring_1) ? var00139_col_current_ring_1 : var00140;
    vec3 var00134 = var00138_col_ring_1;
    vec3 var00132_col_ring_2 = bool(var00075_cond_valid_ring_2) ? var00133_col_current_ring_2 : var00134;
    vec3 var00128 = var00132_col_ring_2;
    vec3 var00126_col_ring_3 = bool(var00059_cond_valid_ring_3) ? var00127_col_current_ring_3 : var00128;
    vec3 var00125 = var00126_col_ring_3;
    vec3 var00121 = var00125;
    vec3 var00119_col_ring_5 = bool(var00042_cond_valid_ring_5) ? var00120_col_current_ring_5 : var00121;
    vec3 var00115 = var00119_col_ring_5;
    vec3 var00113_col_ring_6 = bool(var00026_cond_valid_ring_6) ? var00114_col_current_ring_6 : var00115;
    vec3 var00112 = var00113_col_ring_6;
    vec3 var00005 = var00112;
    vec3 var00002_col_ring_8 = bool(var00003_cond_valid_ring_8) ? var00004_col_current_ring_8 : var00005;
    vec3 var00001 = var00002_col_ring_8;
    vec3 var00000 = var00001;
    
        fragColor = vec4(var00000, 1.0);
        return;
    }