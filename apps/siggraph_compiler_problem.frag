
#define width 960.
#define height 960.

    

// If true, aspect ratio in the original optimization will be preserved, but extra space outside original FOV might reveal artifact.
// If false, FOV is the same as original optimization, but aspect ratio will not be preserved.
#define preserve_aspect_ratio true

// Smaller factor will zoom in the rendering, larger factor will zoom out
float scale_factor = 1.;


// Number of iterations for each raymarching loop
                        
#define _raymarching_iter_0 64
                            

#define X float[](3.2996790545941474e-05, 0.00149179553159809, 1.7404477861110574, 3.14042230690509, -0.0008004511756364425, -0.05055347308924636, -0.5269360525412076, 0.36059339550596925, 0.7715495054525564, 0.09249969780045049, 0.18905217484646403, 0.4004187597280134, 0.21416061328234656, 0.13518932592687638, 1.0147116751732517, 4.028066300761313, 0.2596849757682277, 4.51350547292311, 0.6113281090540534, -0.18520407725053492, 0.28103413605071653, 0.7979866866452583, 1.4919232525496855, 0.38150272860628603, 0.6688595708714182, 0.6609471133243447, 0.060730880312376695, 0.18987583703508507, 0.11804496708169367, 0.41556921417270193, 0.6717356374832354, 0.21994495986012705, 0.6475981528444147, 0.4109814132747472, 0.52788606052313, 0.3123903691003913, 0.1876594887880311, 0.16486372000757413, 0.4860704351927222, 0.3029196392951568, 0.7854949367963268, 0.4809728493228009, 0.2839854436964951)


#define origin_x_idx 0
float origin_x = X[origin_x_idx];

#define origin_y_idx 1
float origin_y = X[origin_y_idx];

#define origin_z_idx 2
float origin_z = X[origin_z_idx];

#define ang_x_idx 3
float ang_x = X[ang_x_idx];

#define ang_y_idx 4
float ang_y = X[ang_y_idx];

#define ang_z_idx 5
float ang_z = X[ang_z_idx];

#define ax_ang_x_idx 6
float ax_ang_x = X[ax_ang_x_idx];

#define ax_ang_y_idx 7
float ax_ang_y = X[ax_ang_y_idx];

#define cone_ang_x_idx 8
float cone_ang_x = X[cone_ang_x_idx];

#define cone_ang_y_idx 9
float cone_ang_y = X[cone_ang_y_idx];

#define cone_ang_z_idx 10
float cone_ang_z = X[cone_ang_z_idx];

#define ellipse_ratio_idx 11
float ellipse_ratio = X[ellipse_ratio_idx];

#define cone_ang_w_idx 12
float cone_ang_w = X[cone_ang_w_idx];

#define d_thre_x_idx 13
float d_thre_x = X[d_thre_x_idx];

#define d_thre_y_idx 14
float d_thre_y = X[d_thre_y_idx];

#define angs_lig0_x_x_idx 15
float angs_lig0_x_x = X[angs_lig0_x_x_idx];

#define angs_lig0_x_y_idx 16
float angs_lig0_x_y = X[angs_lig0_x_y_idx];

#define angs_lig0_y_x_idx 17
float angs_lig0_y_x = X[angs_lig0_y_x_idx];

#define angs_lig0_y_y_idx 18
float angs_lig0_y_y = X[angs_lig0_y_y_idx];

#define pos_lig1_x_x_idx 19
float pos_lig1_x_x = X[pos_lig1_x_x_idx];

#define pos_lig1_x_y_idx 20
float pos_lig1_x_y = X[pos_lig1_x_y_idx];

#define pos_lig1_x_z_idx 21
float pos_lig1_x_z = X[pos_lig1_x_z_idx];

#define pos_lig1_y_x_idx 22
float pos_lig1_y_x = X[pos_lig1_y_x_idx];

#define pos_lig1_y_y_idx 23
float pos_lig1_y_y = X[pos_lig1_y_y_idx];

#define pos_lig1_y_z_idx 24
float pos_lig1_y_z = X[pos_lig1_y_z_idx];

#define amb_x_x_idx 25
float amb_x_x = X[amb_x_x_idx];

#define amb_x_y_idx 26
float amb_x_y = X[amb_x_y_idx];

#define amb_x_z_idx 27
float amb_x_z = X[amb_x_z_idx];

#define amb_y_x_idx 28
float amb_y_x = X[amb_y_x_idx];

#define amb_y_y_idx 29
float amb_y_y = X[amb_y_y_idx];

#define amb_y_z_idx 30
float amb_y_z = X[amb_y_z_idx];

#define kd0_x_x_idx 31
float kd0_x_x = X[kd0_x_x_idx];

#define kd0_x_y_idx 32
float kd0_x_y = X[kd0_x_y_idx];

#define kd0_x_z_idx 33
float kd0_x_z = X[kd0_x_z_idx];

#define kd0_y_x_idx 34
float kd0_y_x = X[kd0_y_x_idx];

#define kd0_y_y_idx 35
float kd0_y_y = X[kd0_y_y_idx];

#define kd0_y_z_idx 36
float kd0_y_z = X[kd0_y_z_idx];

#define kd1_x_x_idx 37
float kd1_x_x = X[kd1_x_x_idx];

#define kd1_x_y_idx 38
float kd1_x_y = X[kd1_x_y_idx];

#define kd1_x_z_idx 39
float kd1_x_z = X[kd1_x_z_idx];

#define kd1_y_x_idx 40
float kd1_y_x = X[kd1_y_x_idx];

#define kd1_y_y_idx 41
float kd1_y_y = X[kd1_y_y_idx];

#define kd1_y_z_idx 42
float kd1_y_z = X[kd1_y_z_idx];

void animate_params() {
}
                    
void animate_raymarching_loop_0_is_fg(
inout bool  raymarching_loop_0_is_fg, 
in float  raymarching_loop_0_t_closest, 
in float  raymarching_loop_0_res0_closest, 
in float  raymarching_loop_0_final_t, 
in float  raymarching_loop_0_final_res0, 
in float  raymarching_loop_0_surface_label){}

void animate_raymarching(
inout float  normalize_final0_surface_normal, 
inout float  normalize_final1_surface_normal, 
inout float  normalize_final2_surface_normal, 
in float  pos_x, 
in float  pos_y, 
in float  pos_z){}


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
    
    float var00012_cos2 = cos(float(ang_y));          
    float var00013_cos3 = cos(float(ang_z));          
    float var00010 = ((var00012_cos2)*(var00013_cos3));
    float var00016 = ((width)*(float(0.5)));          
    float var00014 = ((current_u)-(var00016));        
    float var00020 = ((var00014)*(var00014));         
    float var00023 = ((height)*(float(0.5)));         
    float var00022 = ((current_v)-(var00023));        
    float var00021 = ((var00022)*(var00022));         
    float var00018 = ((var00020)+(var00021));         
    float var00019 = ((var00016)*(var00016));         
    float var00017_rd_norm2 = ((var00018)+(var00019));
    float var00015_sqrt = sqrt(var00017_rd_norm2);    
    float var00011_raw_rd0 = ((var00014)/(var00015_sqrt));
    float var00008 = ((var00010)*(var00011_raw_rd0)); 
    float var00030_cos1 = cos(float(ang_x));          
    float var00028 = (-(float(var00030_cos1)));       
    float var00029_sin3 = sin(float(ang_z));          
    float var00026 = ((var00028)*(var00029_sin3));    
    float var00032_sin1 = sin(float(ang_x));          
    float var00033_sin2 = sin(float(ang_y));          
    float var00031 = ((var00032_sin1)*(var00033_sin2));
    float var00027 = ((var00031)*(var00013_cos3));    
    float var00024 = ((var00026)+(var00027));         
    float var00025_raw_rd1 = ((var00022)/(var00015_sqrt));
    float var00009 = ((var00024)*(var00025_raw_rd1)); 
    float var00006 = ((var00008)+(var00009));         
    float var00036 = ((var00032_sin1)*(var00029_sin3));
    float var00038 = ((var00030_cos1)*(var00033_sin2));
    float var00037 = ((var00038)*(var00013_cos3));    
    float var00034 = ((var00036)+(var00037));         
    float var00035_raw_rd2 = ((var00016)/(var00015_sqrt));
    float var00007 = ((var00034)*(var00035_raw_rd2)); 
    float var00005_rd0 = ((var00006)+(var00007));     
    float var00044 = ((var00012_cos2)*(var00029_sin3));
    float var00042 = ((var00044)*(var00011_raw_rd0)); 
    float var00046 = ((var00030_cos1)*(var00013_cos3));
    float var00047 = ((var00031)*(var00029_sin3));    
    float var00045 = ((var00046)+(var00047));         
    float var00043 = ((var00045)*(var00025_raw_rd1)); 
    float var00040 = ((var00042)+(var00043));         
    float var00051 = (-(float(var00032_sin1)));       
    float var00049 = ((var00051)*(var00013_cos3));    
    float var00050 = ((var00038)*(var00029_sin3));    
    float var00048 = ((var00049)+(var00050));         
    float var00041 = ((var00048)*(var00035_raw_rd2)); 
    float var00039_rd1 = ((var00040)+(var00041));     
    float var00057 = (-(float(var00033_sin2)));       
    float var00055 = ((var00057)*(var00011_raw_rd0)); 
    float var00058 = ((var00032_sin1)*(var00012_cos2));
    float var00056 = ((var00058)*(var00025_raw_rd1)); 
    float var00053 = ((var00055)+(var00056));         
    float var00059 = ((var00030_cos1)*(var00012_cos2));
    float var00054 = ((var00059)*(var00035_raw_rd2)); 
    float var00052_rd2 = ((var00053)+(var00054));     
    float var00061_sin_theta = sin(float(ax_ang_x));  
    float var00062_cos_phi = cos(float(ax_ang_y));    
    float var00060_ax0 = ((var00061_sin_theta)*(var00062_cos_phi));
    float var00064_cos_theta = cos(float(ax_ang_x));  
    float var00063_ax1 = ((var00064_cos_theta)*(var00062_cos_phi));
    float var00065_ax2 = sin(float(ax_ang_y));        
    float var00066_cos_cone_ang = cos(float(cone_ang_w));
    float var00068_cone_v2_1 = sin(float(cone_ang_x));
    float var00069_cone_v0_2 = cos(float(cone_ang_y));
    float var00067_cone_v1_0 = ((var00068_cone_v2_1)*(var00069_cone_v0_2));
    float var00071_cos_cone_theta = cos(float(cone_ang_x));
    float var00070_cone_v1_1 = ((var00071_cos_cone_theta)*(var00069_cone_v0_2));
    float var00073_sin_cone_phi = sin(float(cone_ang_y));
    float var00072_cone_v1_2 = (-(float(var00073_sin_cone_phi)));
    float var00074_cos_cone_alpha = cos(float(cone_ang_z));
    float var00075_cone_v2_0 = (-(float(var00071_cos_cone_theta)));
    float var00076_sin_cone_alpha = sin(float(cone_ang_z));
    float var00077_sin_cone_ang = sin(float(cone_ang_w));
    float var00078_cone_v0_0 = ((var00068_cone_v2_1)*(var00073_sin_cone_phi));
    float var00079_cone_v0_1 = ((var00071_cos_cone_theta)*(var00073_sin_cone_phi));
    float _t_raymarching_0 = float(0);
    float _t_closest_raymarching_0 = _t_raymarching_0;
    float _res0_closest_raymarching_0 = float(10.0);
    float[6] var00004;
    for (int raymarching_loop_i = 0; raymarching_loop_i < _raymarching_iter_0; raymarching_loop_i++) {
        float var00100 = ((var00005_rd0)*(_t_raymarching_0));
        float var00099 = ((float(origin_x))+(var00100));  
        float var00098_deriv_d2_pos0_x = ((var00099)*(float(1.0)));
        float var00097 = ((var00098_deriv_d2_pos0_x)*(var00060_ax0));
        float var00095 = ((float(0))+(var00097));         
        float var00103 = ((var00039_rd1)*(_t_raymarching_0));
        float var00102 = ((float(origin_y))+(var00103));  
        float var00101_deriv_d2_pos1_x = ((var00102)*(float(1.0)));
        float var00096 = ((var00101_deriv_d2_pos1_x)*(var00063_ax1));
        float var00093 = ((var00095)+(var00096));         
        float var00106 = ((var00052_rd2)*(_t_raymarching_0));
        float var00105 = ((float(origin_z))+(var00106));  
        float var00104_deriv_d2_pos2_x = ((var00105)*(float(1.0)));
        float var00094 = ((var00104_deriv_d2_pos2_x)*(var00065_ax2));
        float var00092 = ((var00093)+(var00094));         
        float var00090_d1_x = ((var00092)-(float(d_thre_x)));
        float var00111_pos0_squared_x = ((var00098_deriv_d2_pos0_x)*(var00098_deriv_d2_pos0_x));
        float var00112_pos1_squared_x = ((var00101_deriv_d2_pos1_x)*(var00101_deriv_d2_pos1_x));
        float var00109 = ((var00111_pos0_squared_x)+(var00112_pos1_squared_x));
        float var00110_pos2_squared_x = ((var00104_deriv_d2_pos2_x)*(var00104_deriv_d2_pos2_x));
        float var00108_dist2_x = ((var00109)+(var00110_pos2_squared_x));
        float var00107_dist_x = sqrt(var00108_dist2_x);   
        float var00091_d2_x = ((var00107_dist_x)-(float(d_thre_y)));
        bool var00089 = ((var00090_d1_x)>=(var00091_d2_x));
        float var00087_select = bool(var00089) ? var00090_d1_x : var00091_d2_x;
        float var00131 = ((var00099)-(float(origin_x)));  
        float var00130_q0_x = ((float(1.0))*(var00131));  
        float var00129 = ((var00130_q0_x)*(var00067_cone_v1_0));
        float var00127 = ((float(0))+(var00129));         
        float var00133 = ((var00102)-(float(origin_y)));  
        float var00132_q1_x = ((float(1.0))*(var00133));  
        float var00128 = ((var00132_q1_x)*(var00070_cone_v1_1));
        float var00125 = ((var00127)+(var00128));         
        float var00135 = ((var00105)-(float(origin_z)));  
        float var00134_q2_x = ((float(1.0))*(var00135));  
        float var00126 = ((var00134_q2_x)*(var00072_cone_v1_2));
        float var00124_q1_x = ((var00125)+(var00126));    
        float var00122 = ((var00124_q1_x)*(var00074_cos_cone_alpha));
        float var00141 = ((var00130_q0_x)*(var00075_cone_v2_0));
        float var00139 = ((float(0))+(var00141));         
        float var00140 = ((var00132_q1_x)*(var00068_cone_v2_1));
        float var00137 = ((var00139)+(var00140));         
        float var00138 = ((var00134_q2_x)*(float(0.0)));  
        float var00136_q2_x = ((var00137)+(var00138));    
        float var00123 = ((var00136_q2_x)*(var00076_sin_cone_alpha));
        float var00121_r1_x = ((var00122)+(var00123));    
        float var00120 = ((var00121_r1_x)*(float(ellipse_ratio)));
        float var00118 = ((var00120)*(var00120));         
        float var00143 = ((var00124_q1_x)*(var00076_sin_cone_alpha));
        float var00144 = ((var00136_q2_x)*(var00074_cos_cone_alpha));
        float var00142_r2_x = ((var00143)+(var00144));    
        float var00119 = ((var00142_r2_x)*(var00142_r2_x));
        float var00117 = ((var00118)+(var00119));         
        float var00116_scaled_dist_x = sqrt(var00117);    
        float var00114 = ((var00066_cos_cone_ang)*(var00116_scaled_dist_x));
        float var00150 = ((var00130_q0_x)*(var00078_cone_v0_0));
        float var00148 = ((float(0))+(var00150));         
        float var00149 = ((var00132_q1_x)*(var00079_cone_v0_1));
        float var00146 = ((var00148)+(var00149));         
        float var00147 = ((var00134_q2_x)*(var00069_cone_v0_2));
        float var00145_q0_x = ((var00146)+(var00147));    
        float var00115 = ((var00077_sin_cone_ang)*(var00145_q0_x));
        float var00113_d3_x = ((var00114)+(var00115));    
        float var00088 = (-(float(var00113_d3_x)));       
        bool var00086 = ((var00087_select)>=(var00088));  
        float var00084_select = bool(var00086) ? var00087_select : var00088;
        float var00163_deriv_d2_pos0_y = ((var00099)*(float(-1)));
        float var00162 = ((var00163_deriv_d2_pos0_y)*(var00060_ax0));
        float var00160 = ((float(0))+(var00162));         
        float var00164_deriv_d2_pos1_y = ((var00102)*(float(-1)));
        float var00161 = ((var00164_deriv_d2_pos1_y)*(var00063_ax1));
        float var00158 = ((var00160)+(var00161));         
        float var00165_deriv_d2_pos2_y = ((var00105)*(float(1)));
        float var00159 = ((var00165_deriv_d2_pos2_y)*(var00065_ax2));
        float var00157 = ((var00158)+(var00159));         
        float var00155_d1_y = ((var00157)-(float(d_thre_x)));
        float var00170_pos0_squared_y = ((var00163_deriv_d2_pos0_y)*(var00163_deriv_d2_pos0_y));
        float var00171_pos1_squared_y = ((var00164_deriv_d2_pos1_y)*(var00164_deriv_d2_pos1_y));
        float var00168 = ((var00170_pos0_squared_y)+(var00171_pos1_squared_y));
        float var00169_pos2_squared_y = ((var00165_deriv_d2_pos2_y)*(var00165_deriv_d2_pos2_y));
        float var00167_dist2_y = ((var00168)+(var00169_pos2_squared_y));
        float var00166_dist_y = sqrt(var00167_dist2_y);   
        float var00156_d2_y = ((var00166_dist_y)-(float(d_thre_y)));
        bool var00154 = ((var00155_d1_y)>=(var00156_d2_y));
        float var00152_select = bool(var00154) ? var00155_d1_y : var00156_d2_y;
        float var00190 = ((var00099)-(float(origin_x)));  
        float var00189_q0_y = ((float(-1))*(var00190));   
        float var00188 = ((var00189_q0_y)*(var00067_cone_v1_0));
        float var00186 = ((float(0))+(var00188));         
        float var00192 = ((var00102)-(float(origin_y)));  
        float var00191_q1_y = ((float(-1))*(var00192));   
        float var00187 = ((var00191_q1_y)*(var00070_cone_v1_1));
        float var00184 = ((var00186)+(var00187));         
        float var00194 = ((var00105)-(float(origin_z)));  
        float var00193_q2_y = ((float(1))*(var00194));    
        float var00185 = ((var00193_q2_y)*(var00072_cone_v1_2));
        float var00183_q1_y = ((var00184)+(var00185));    
        float var00181 = ((var00183_q1_y)*(var00074_cos_cone_alpha));
        float var00200 = ((var00189_q0_y)*(var00075_cone_v2_0));
        float var00198 = ((float(0))+(var00200));         
        float var00199 = ((var00191_q1_y)*(var00068_cone_v2_1));
        float var00196 = ((var00198)+(var00199));         
        float var00197 = ((var00193_q2_y)*(float(0.0)));  
        float var00195_q2_y = ((var00196)+(var00197));    
        float var00182 = ((var00195_q2_y)*(var00076_sin_cone_alpha));
        float var00180_r1_y = ((var00181)+(var00182));    
        float var00179 = ((var00180_r1_y)*(float(ellipse_ratio)));
        float var00177 = ((var00179)*(var00179));         
        float var00202 = ((var00183_q1_y)*(var00076_sin_cone_alpha));
        float var00203 = ((var00195_q2_y)*(var00074_cos_cone_alpha));
        float var00201_r2_y = ((var00202)+(var00203));    
        float var00178 = ((var00201_r2_y)*(var00201_r2_y));
        float var00176 = ((var00177)+(var00178));         
        float var00175_scaled_dist_y = sqrt(var00176);    
        float var00173 = ((var00066_cos_cone_ang)*(var00175_scaled_dist_y));
        float var00209 = ((var00189_q0_y)*(var00078_cone_v0_0));
        float var00207 = ((float(0))+(var00209));         
        float var00208 = ((var00191_q1_y)*(var00079_cone_v0_1));
        float var00205 = ((var00207)+(var00208));         
        float var00206 = ((var00193_q2_y)*(var00069_cone_v0_2));
        float var00204_q0_y = ((var00205)+(var00206));    
        float var00174 = ((var00077_sin_cone_ang)*(var00204_q0_y));
        float var00172_d3_y = ((var00173)+(var00174));    
        float var00153 = (-(float(var00172_d3_y)));       
        bool var00151 = ((var00152_select)>=(var00153));  
        float var00085_select = bool(var00151) ? var00152_select : var00153;
        bool var00083 = ((var00084_select)<=(var00085_select));
        float var00081_select = bool(var00083) ? var00084_select : var00085_select;
        bool var00211 = ((var00081_select)<(_res0_closest_raymarching_0));
        float var00210_select = bool(var00211) ? _t_raymarching_0 : _t_closest_raymarching_0;
        float var00082 = ((float(0.0004))*(var00210_select));
        bool var00080 = ((var00081_select)<(var00082));   
        float var00212_select = bool(var00211) ? var00081_select : _res0_closest_raymarching_0;
        float var00213 = ((_t_raymarching_0)+(var00081_select));
        float var00216 = ((var00085_select)-(var00084_select));
        bool var00215_cond_xy = ((var00216)>(float(0)));  
        float var00214_combined_res1 = bool(var00215_cond_xy) ? float(0.0) : float(1.0);
        float[6] var00217 = float[](float(var00080), float(var00210_select), float(var00212_select), float(var00213), float(var00081_select), float(var00214_combined_res1));
        var00004 = var00217;
        _t_closest_raymarching_0 = var00004[1];
        _res0_closest_raymarching_0 = var00004[2];
        _t_raymarching_0 = var00004[3];
    }
    bool var00003_raymarching_loop_0_is_fg = bool(var00004[int(float(0))]);
    bool  var00001_raymarching_loop_0_is_fg = var00003_raymarching_loop_0_is_fg;
    float var00218_raymarching_loop_0_t_closest = float(var00004[int(float(1))]);
    float var00219_raymarching_loop_0_res0_closest = float(var00004[int(float(2))]);
    float var00220_raymarching_loop_0_final_t = float(var00004[int(float(3))]);
    float var00221_raymarching_loop_0_final_res0 = float(var00004[int(float(4))]);
    float var00222_raymarching_loop_0_surface_label = float(var00004[int(float(5))]);
    animate_raymarching_loop_0_is_fg(var00001_raymarching_loop_0_is_fg, var00218_raymarching_loop_0_t_closest, var00219_raymarching_loop_0_res0_closest, var00220_raymarching_loop_0_final_t, var00221_raymarching_loop_0_final_res0, var00222_raymarching_loop_0_surface_label);
    float var00261 = ((var00005_rd0)*(var00218_raymarching_loop_0_t_closest));
    float var00260 = ((float(origin_x))+(var00261));  
    float var00259_deriv_d2_pos0_y = ((var00260)*(float(-1)));
    float var00257 = ((var00259_deriv_d2_pos0_y)*(var00060_ax0));
    float var00264 = ((var00039_rd1)*(var00218_raymarching_loop_0_t_closest));
    float var00263 = ((float(origin_y))+(var00264));  
    float var00262_deriv_d2_pos1_y = ((var00263)*(float(-1)));
    float var00258 = ((var00262_deriv_d2_pos1_y)*(var00063_ax1));
    float var00255 = ((var00257)+(var00258));         
    float var00266 = ((var00052_rd2)*(var00218_raymarching_loop_0_t_closest));
    float var00265 = ((float(origin_z))+(var00266));  
    float var00256 = ((var00265)*(var00065_ax2));     
    float var00254 = ((var00255)+(var00256));         
    float var00252_d1_y = ((var00254)-(float(d_thre_x)));
    float var00271_pos0_squared_y = ((var00259_deriv_d2_pos0_y)*(var00259_deriv_d2_pos0_y));
    float var00272_pos1_squared_y = ((var00262_deriv_d2_pos1_y)*(var00262_deriv_d2_pos1_y));
    float var00269 = ((var00271_pos0_squared_y)+(var00272_pos1_squared_y));
    float var00270_pos2_squared_y = ((var00265)*(var00265));
    float var00268_dist2_y = ((var00269)+(var00270_pos2_squared_y));
    float var00267_dist_y = sqrt(var00268_dist2_y);   
    float var00253_d2_y = ((var00267_dist_y)-(float(d_thre_y)));
    bool var00251 = ((var00252_d1_y)>=(var00253_d2_y));
    float var00249_select = bool(var00251) ? var00252_d1_y : var00253_d2_y;
    float var00290 = ((var00260)-(float(origin_x)));  
    float var00289_q0_y = ((float(-1))*(var00290));   
    float var00287 = ((var00289_q0_y)*(var00067_cone_v1_0));
    float var00292 = ((var00263)-(float(origin_y)));  
    float var00291_q1_y = ((float(-1))*(var00292));   
    float var00288 = ((var00291_q1_y)*(var00070_cone_v1_1));
    float var00285 = ((var00287)+(var00288));         
    float var00293 = ((var00265)-(float(origin_z)));  
    float var00286 = ((var00293)*(var00072_cone_v1_2));
    float var00284_q1_y = ((var00285)+(var00286));    
    float var00282 = ((var00284_q1_y)*(var00074_cos_cone_alpha));
    float var00295 = ((var00289_q0_y)*(var00075_cone_v2_0));
    float var00296 = ((var00291_q1_y)*(var00068_cone_v2_1));
    float var00294 = ((var00295)+(var00296));         
    float var00283 = ((var00294)*(var00076_sin_cone_alpha));
    float var00281_r1_y = ((var00282)+(var00283));    
    float var00280 = ((var00281_r1_y)*(float(ellipse_ratio)));
    float var00278 = ((var00280)*(var00280));         
    float var00298 = ((var00284_q1_y)*(var00076_sin_cone_alpha));
    float var00299 = ((var00294)*(var00074_cos_cone_alpha));
    float var00297_r2_y = ((var00298)+(var00299));    
    float var00279 = ((var00297_r2_y)*(var00297_r2_y));
    float var00277 = ((var00278)+(var00279));         
    float var00276_scaled_dist_y = sqrt(var00277);    
    float var00274 = ((var00066_cos_cone_ang)*(var00276_scaled_dist_y));
    float var00303 = ((var00289_q0_y)*(var00078_cone_v0_0));
    float var00304 = ((var00291_q1_y)*(var00079_cone_v0_1));
    float var00301 = ((var00303)+(var00304));         
    float var00302 = ((var00293)*(var00069_cone_v0_2));
    float var00300_q0_y = ((var00301)+(var00302));    
    float var00275 = ((var00077_sin_cone_ang)*(var00300_q0_y));
    float var00273_d3_y = ((var00274)+(var00275));    
    float var00250 = (-(float(var00273_d3_y)));       
    bool var00248 = ((var00249_select)>=(var00250));  
    float var00246_select = bool(var00248) ? var00249_select : var00250;
    float var00313 = ((var00260)*(var00060_ax0));     
    float var00314 = ((var00263)*(var00063_ax1));     
    float var00312 = ((var00313)+(var00314));         
    float var00311 = ((var00312)+(var00256));         
    float var00309_d1_x = ((var00311)-(float(d_thre_x)));
    float var00318_pos0_squared_x = ((var00260)*(var00260));
    float var00319_pos1_squared_x = ((var00263)*(var00263));
    float var00317 = ((var00318_pos0_squared_x)+(var00319_pos1_squared_x));
    float var00316_dist2_x = ((var00317)+(var00270_pos2_squared_y));
    float var00315_dist_x = sqrt(var00316_dist2_x);   
    float var00310_d2_x = ((var00315_dist_x)-(float(d_thre_y)));
    bool var00308 = ((var00309_d1_x)>=(var00310_d2_x));
    float var00306_select = bool(var00308) ? var00309_d1_x : var00310_d2_x;
    float var00333 = ((var00290)*(var00067_cone_v1_0));
    float var00334 = ((var00292)*(var00070_cone_v1_1));
    float var00332 = ((var00333)+(var00334));         
    float var00331_q1_x = ((var00332)+(var00286));    
    float var00329 = ((var00331_q1_x)*(var00074_cos_cone_alpha));
    float var00336 = ((var00290)*(var00075_cone_v2_0));
    float var00337 = ((var00292)*(var00068_cone_v2_1));
    float var00335 = ((var00336)+(var00337));         
    float var00330 = ((var00335)*(var00076_sin_cone_alpha));
    float var00328_r1_x = ((var00329)+(var00330));    
    float var00327 = ((var00328_r1_x)*(float(ellipse_ratio)));
    float var00325 = ((var00327)*(var00327));         
    float var00339 = ((var00331_q1_x)*(var00076_sin_cone_alpha));
    float var00340 = ((var00335)*(var00074_cos_cone_alpha));
    float var00338_r2_x = ((var00339)+(var00340));    
    float var00326 = ((var00338_r2_x)*(var00338_r2_x));
    float var00324 = ((var00325)+(var00326));         
    float var00323_scaled_dist_x = sqrt(var00324);    
    float var00321 = ((var00066_cos_cone_ang)*(var00323_scaled_dist_x));
    float var00343 = ((var00290)*(var00078_cone_v0_0));
    float var00344 = ((var00292)*(var00079_cone_v0_1));
    float var00342 = ((var00343)+(var00344));         
    float var00341_q0_x = ((var00342)+(var00302));    
    float var00322 = ((var00077_sin_cone_ang)*(var00341_q0_x));
    float var00320_d3_x = ((var00321)+(var00322));    
    float var00307 = (-(float(var00320_d3_x)));       
    bool var00305 = ((var00306_select)>=(var00307));  
    float var00247_select = bool(var00305) ? var00306_select : var00307;
    float var00245 = ((var00246_select)-(var00247_select));
    bool var00242_cond_xy = ((var00245)>(float(0)));  
    bool var00345_cond0_x = ((var00309_d1_x)>(var00310_d2_x));
    float var00243_deriv_t_shell_pos0_x = bool(var00345_cond0_x) ? var00060_ax0 : var00260;
    bool var00347_cond0_y = ((var00252_d1_y)>(var00253_d2_y));
    float var00346_deriv_t_shell_pos0_y = bool(var00347_cond0_y) ? var00060_ax0 : var00259_deriv_d2_pos0_y;
    float var00244 = (-(float(var00346_deriv_t_shell_pos0_y)));
    float var00240_deriv_sdf0 = bool(var00242_cond_xy) ? var00243_deriv_t_shell_pos0_x : var00244;
    float var00352_normalize_in_squared0_surface_normal = ((var00240_deriv_sdf0)*(var00240_deriv_sdf0));
    float var00355_deriv_t_shell_pos1_x = bool(var00345_cond0_x) ? var00063_ax1 : var00263;
    float var00357_deriv_t_shell_pos1_y = bool(var00347_cond0_y) ? var00063_ax1 : var00262_deriv_d2_pos1_y;
    float var00356 = (-(float(var00357_deriv_t_shell_pos1_y)));
    float var00354_deriv_sdf1 = bool(var00242_cond_xy) ? var00355_deriv_t_shell_pos1_x : var00356;
    float var00353_normalize_in_squared1_surface_normal = ((var00354_deriv_sdf1)*(var00354_deriv_sdf1));
    float var00350 = ((var00352_normalize_in_squared0_surface_normal)+(var00353_normalize_in_squared1_surface_normal));
    float var00359_deriv_t_shell_pos2_x = bool(var00345_cond0_x) ? var00065_ax2 : var00265;
    float var00360_deriv_t_shell_pos2_y = bool(var00347_cond0_y) ? var00065_ax2 : var00265;
    float var00358_deriv_sdf2 = bool(var00242_cond_xy) ? var00359_deriv_t_shell_pos2_x : var00360_deriv_t_shell_pos2_y;
    float var00351_normalize_in_squared2_surface_normal = ((var00358_deriv_sdf2)*(var00358_deriv_sdf2));
    float var00349_normalize_norm2_surface_normal = ((var00350)+(var00351_normalize_in_squared2_surface_normal));
    float var00348_normalize_inv_norm_surface_normal_inv = sqrt(var00349_normalize_norm2_surface_normal);
    float var00241_normalize_inv_norm_surface_normal = ((float(1))/(var00348_normalize_inv_norm_surface_normal_inv));
    float var00239_normalize_final0_surface_normal = ((var00240_deriv_sdf0)*(var00241_normalize_inv_norm_surface_normal));
    float  var00237_normalize_final0_surface_normal = var00239_normalize_final0_surface_normal;
    float var00362_normalize_final1_surface_normal = ((var00354_deriv_sdf1)*(var00241_normalize_inv_norm_surface_normal));
    float  var00361_normalize_final1_surface_normal = var00362_normalize_final1_surface_normal;
    float var00364_normalize_final2_surface_normal = ((var00358_deriv_sdf2)*(var00241_normalize_inv_norm_surface_normal));
    float  var00363_normalize_final2_surface_normal = var00364_normalize_final2_surface_normal;
    float var00365_pos_x = ((float(origin_x))+(var00261));
    float var00366_pos_y = ((float(origin_y))+(var00264));
    float var00367_pos_z = ((float(origin_z))+(var00266));
    animate_raymarching(var00237_normalize_final0_surface_normal, var00361_normalize_final1_surface_normal, var00363_normalize_final2_surface_normal, var00365_pos_x, var00366_pos_y, var00367_pos_z);
    float var00368_sin_theta_lig0_x = sin(float(angs_lig0_x_x));
    float var00369_cos_phi_lig0_x = cos(float(angs_lig0_x_y));
    float var00238_dir_lig0_x0 = ((var00368_sin_theta_lig0_x)*(var00369_cos_phi_lig0_x));
    float var00235 = ((var00237_normalize_final0_surface_normal)*(var00238_dir_lig0_x0));
    float var00371_cos_theta_lig0_x = cos(float(angs_lig0_x_x));
    float var00370_dir_lig0_x1 = ((var00371_cos_theta_lig0_x)*(var00369_cos_phi_lig0_x));
    float var00236 = ((var00361_normalize_final1_surface_normal)*(var00370_dir_lig0_x1));
    float var00233 = ((var00235)+(var00236));         
    float var00372_dir_lig0_x2 = sin(float(angs_lig0_x_y));
    float var00234 = ((var00363_normalize_final2_surface_normal)*(var00372_dir_lig0_x2));
    float var00232_dot_lig0_x = ((var00233)+(var00234));
    bool var00231_cond_dif0_x = ((var00232_dot_lig0_x)>(float(0)));
    float var00230_dif0_x_sc = bool(var00231_cond_dif0_x) ? var00232_dot_lig0_x : float(0);
    float var00229 = ((var00230_dif0_x_sc)*(float(kd0_x_x)));
    float var00227 = ((float(amb_x_x))+(var00229));   
    float var00381_dir_lig1_x_diff0 = ((float(pos_lig1_x_x))-(var00260));
    float var00386_normalize_in_squared0_dir_lig1_x_diff = ((var00381_dir_lig1_x_diff0)*(var00381_dir_lig1_x_diff0));
    float var00388_dir_lig1_x_diff1 = ((float(pos_lig1_x_y))-(var00263));
    float var00387_normalize_in_squared1_dir_lig1_x_diff = ((var00388_dir_lig1_x_diff1)*(var00388_dir_lig1_x_diff1));
    float var00384 = ((var00386_normalize_in_squared0_dir_lig1_x_diff)+(var00387_normalize_in_squared1_dir_lig1_x_diff));
    float var00389_dir_lig1_x_diff2 = ((float(pos_lig1_x_z))-(var00265));
    float var00385_normalize_in_squared2_dir_lig1_x_diff = ((var00389_dir_lig1_x_diff2)*(var00389_dir_lig1_x_diff2));
    float var00383_normalize_norm2_dir_lig1_x_diff = ((var00384)+(var00385_normalize_in_squared2_dir_lig1_x_diff));
    float var00382_normalize_inv_norm_dir_lig1_x_diff_inv = sqrt(var00383_normalize_norm2_dir_lig1_x_diff);
    float var00380_dir_lig1_x0 = ((var00381_dir_lig1_x_diff0)/(var00382_normalize_inv_norm_dir_lig1_x_diff_inv));
    float var00378 = ((var00237_normalize_final0_surface_normal)*(var00380_dir_lig1_x0));
    float var00390_dir_lig1_x1 = ((var00388_dir_lig1_x_diff1)/(var00382_normalize_inv_norm_dir_lig1_x_diff_inv));
    float var00379 = ((var00361_normalize_final1_surface_normal)*(var00390_dir_lig1_x1));
    float var00376 = ((var00378)+(var00379));         
    float var00391_dir_lig1_x2 = ((var00389_dir_lig1_x_diff2)/(var00382_normalize_inv_norm_dir_lig1_x_diff_inv));
    float var00377 = ((var00363_normalize_final2_surface_normal)*(var00391_dir_lig1_x2));
    float var00375_dot_lig1_x = ((var00376)+(var00377));
    bool var00374_cond_dif1_x = ((var00375_dot_lig1_x)>(float(0)));
    float var00373_dif1_x_sc = bool(var00374_cond_dif1_x) ? var00375_dot_lig1_x : float(0);
    float var00228 = ((var00373_dif1_x_sc)*(float(kd1_x_x)));
    float var00225_col_x0 = ((var00227)+(var00228));  
    float var00226 = ((float(1.0))-(var00222_raymarching_loop_0_surface_label));
    float var00223 = ((var00225_col_x0)*(var00226));  
    float var00404_sin_theta_lig0_y = sin(float(angs_lig0_y_x));
    float var00405_cos_phi_lig0_y = cos(float(angs_lig0_y_y));
    float var00403_dir_lig0_y0 = ((var00404_sin_theta_lig0_y)*(var00405_cos_phi_lig0_y));
    float var00401 = ((var00237_normalize_final0_surface_normal)*(var00403_dir_lig0_y0));
    float var00407_cos_theta_lig0_y = cos(float(angs_lig0_y_x));
    float var00406_dir_lig0_y1 = ((var00407_cos_theta_lig0_y)*(var00405_cos_phi_lig0_y));
    float var00402 = ((var00361_normalize_final1_surface_normal)*(var00406_dir_lig0_y1));
    float var00399 = ((var00401)+(var00402));         
    float var00408_dir_lig0_y2 = sin(float(angs_lig0_y_y));
    float var00400 = ((var00363_normalize_final2_surface_normal)*(var00408_dir_lig0_y2));
    float var00398_dot_lig0_y = ((var00399)+(var00400));
    bool var00397_cond_dif0_y = ((var00398_dot_lig0_y)>(float(0)));
    float var00396_dif0_y_sc = bool(var00397_cond_dif0_y) ? var00398_dot_lig0_y : float(0);
    float var00395 = ((var00396_dif0_y_sc)*(float(kd0_y_x)));
    float var00393 = ((float(amb_y_x))+(var00395));   
    float var00417_dir_lig1_y_diff0 = ((float(pos_lig1_y_x))-(var00260));
    float var00422_normalize_in_squared0_dir_lig1_y_diff = ((var00417_dir_lig1_y_diff0)*(var00417_dir_lig1_y_diff0));
    float var00424_dir_lig1_y_diff1 = ((float(pos_lig1_y_y))-(var00263));
    float var00423_normalize_in_squared1_dir_lig1_y_diff = ((var00424_dir_lig1_y_diff1)*(var00424_dir_lig1_y_diff1));
    float var00420 = ((var00422_normalize_in_squared0_dir_lig1_y_diff)+(var00423_normalize_in_squared1_dir_lig1_y_diff));
    float var00425_dir_lig1_y_diff2 = ((float(pos_lig1_y_z))-(var00265));
    float var00421_normalize_in_squared2_dir_lig1_y_diff = ((var00425_dir_lig1_y_diff2)*(var00425_dir_lig1_y_diff2));
    float var00419_normalize_norm2_dir_lig1_y_diff = ((var00420)+(var00421_normalize_in_squared2_dir_lig1_y_diff));
    float var00418_normalize_inv_norm_dir_lig1_y_diff_inv = sqrt(var00419_normalize_norm2_dir_lig1_y_diff);
    float var00416_dir_lig1_y0 = ((var00417_dir_lig1_y_diff0)/(var00418_normalize_inv_norm_dir_lig1_y_diff_inv));
    float var00414 = ((var00237_normalize_final0_surface_normal)*(var00416_dir_lig1_y0));
    float var00426_dir_lig1_y1 = ((var00424_dir_lig1_y_diff1)/(var00418_normalize_inv_norm_dir_lig1_y_diff_inv));
    float var00415 = ((var00361_normalize_final1_surface_normal)*(var00426_dir_lig1_y1));
    float var00412 = ((var00414)+(var00415));         
    float var00427_dir_lig1_y2 = ((var00425_dir_lig1_y_diff2)/(var00418_normalize_inv_norm_dir_lig1_y_diff_inv));
    float var00413 = ((var00363_normalize_final2_surface_normal)*(var00427_dir_lig1_y2));
    float var00411_dot_lig1_y = ((var00412)+(var00413));
    bool var00410_cond_dif1_y = ((var00411_dot_lig1_y)>(float(0)));
    float var00409_dif1_y_sc = bool(var00410_cond_dif1_y) ? var00411_dot_lig1_y : float(0);
    float var00394 = ((var00409_dif1_y_sc)*(float(kd1_y_x)));
    float var00392_col_y0 = ((var00393)+(var00394));  
    float var00224 = ((var00392_col_y0)*(var00222_raymarching_loop_0_surface_label));
    float var00002_col_obj0 = ((var00223)+(var00224));
    float var00000_col_R = bool(var00001_raymarching_loop_0_is_fg) ? var00002_col_obj0 : float(1.0);
    float var00435 = ((var00230_dif0_x_sc)*(float(kd0_x_y)));
    float var00433 = ((float(amb_x_y))+(var00435));   
    float var00434 = ((var00373_dif1_x_sc)*(float(kd1_x_y)));
    float var00432_col_x1 = ((var00433)+(var00434));  
    float var00430 = ((var00432_col_x1)*(var00226));  
    float var00439 = ((var00396_dif0_y_sc)*(float(kd0_y_y)));
    float var00437 = ((float(amb_y_y))+(var00439));   
    float var00438 = ((var00409_dif1_y_sc)*(float(kd1_y_y)));
    float var00436_col_y1 = ((var00437)+(var00438));  
    float var00431 = ((var00436_col_y1)*(var00222_raymarching_loop_0_surface_label));
    float var00429_col_obj1 = ((var00430)+(var00431));
    float var00428_col_G = bool(var00001_raymarching_loop_0_is_fg) ? var00429_col_obj1 : float(1.0);
    float var00447 = ((var00230_dif0_x_sc)*(float(kd0_x_z)));
    float var00445 = ((float(amb_x_z))+(var00447));   
    float var00446 = ((var00373_dif1_x_sc)*(float(kd1_x_z)));
    float var00444_col_x2 = ((var00445)+(var00446));  
    float var00442 = ((var00444_col_x2)*(var00226));  
    float var00451 = ((var00396_dif0_y_sc)*(float(kd0_y_z)));
    float var00449 = ((float(amb_y_z))+(var00451));   
    float var00450 = ((var00409_dif1_y_sc)*(float(kd1_y_z)));
    float var00448_col_y2 = ((var00449)+(var00450));  
    float var00443 = ((var00448_col_y2)*(var00222_raymarching_loop_0_surface_label));
    float var00441_col_obj2 = ((var00442)+(var00443));
    float var00440_col_B = bool(var00001_raymarching_loop_0_is_fg) ? var00441_col_obj2 : float(1.0);
    vec3 var00452 = vec3(float(var00000_col_R), float(var00428_col_G), float(var00440_col_B));
    
        fragColor = vec4(var00452, 1.0);
        return;
    }