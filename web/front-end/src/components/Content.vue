<template>
  <div id="Content">
    <el-dialog
      title="模型测试中"
      :visible.sync="dialogTableVisible"
      :show-close="false"
      :close-on-press-escape="false"
      :append-to-body="true"
      :close-on-click-modal="false"
      :center="true"
    >
      <el-progress :percentage="percentage"></el-progress>
      <span slot="footer">请耐心等待<i class="el-icon-loading"></i></span>
    </el-dialog>

    <div id="CT">
      <div id="CT_image">
        <el-card
          id="CT_image_1"
          class="box-card"
          style="
            border-radius: 8px;
            width: 800px;
            height: 540px;
            margin-bottom: -30px;
          "
        >


        <div slot="header" class="clearfix">
          <el-upload
            class="upload-demo"
            action="#"
            accept=".pt"
            :http-request="httpRequestModel"      
            :before-remove="beforeRemove"
            :limit="1"
            :on-exceed="handleExceed"
            :file-list="fileListModel"
            >
            <el-button size="small" type="primary" icon="el-icon-upload" :disabled="showResult">上传模型</el-button>
            <div slot="tip" class="el-upload__tip">仅支持上传pt格式模型（若选择不上传，则使用系统自带的模型）</div>
          </el-upload>
          
        </div>

          <div class="demo-image__preview1">
            <div
              v-loading="loading"
              element-loading-text="上传图片中"
              element-loading-spinner="el-icon-loading"
            >
              <el-image
                :src="url_1"
                class="image_1"
                :preview-src-list="srcList"
                style="border-radius: 3px 3px 0 0"
              >
                <div slot="error">
                  <div slot="placeholder" class="error">
                    <el-upload
                      class="upload-demo"
                      action="#"
                      accept=".jpg, .png, .zip"
                      :before-remove="beforeRemoveImage"
                      :http-request="httpRequestImage" 
                      :limit="1"
                      :on-exceed="handleExceedImage"
                      :file-list="fileListImage">
                      
                      <el-tooltip class="item" effect="dark" content="若不传则使用系统自带数据集" placement="top">
                        <el-button size="small" type="primary" icon="el-icon-upload" >上传图像</el-button>
                      </el-tooltip>

                      <div slot="tip" class="el-upload__tip">仅支持上传jpg/png或多张压缩成zip格式</div>
                    </el-upload>
                  </div>
                </div>
              </el-image>
            </div>
            <div class="img_info_1" style="border-radius: 0 0 5px 5px">
              <span style="color: white; letter-spacing: 6px">{{showResult ? '干净效果' : '数据集' }}</span>
            </div>
          </div>
          
          <div class="demo-image__preview2">
            <div
              v-loading="loading"
              element-loading-text="处理中,请耐心等待"
              element-loading-spinner="el-icon-loading"
            >
              <el-image
                :src="url_2"
                class="image_1"
                :preview-src-list="srcList1"
                style="border-radius: 3px 3px 0 0"
              >
                <div slot="error">
                  <div slot="placeholder" class="error">{{ wait_return }}</div>
                </div>
              </el-image>
            </div>
            <div class="img_info_1" style="border-radius: 0 0 5px 5px">
              <span style="color: white; letter-spacing: 4px">{{ showResult ? '扰动效果' : '检测结果'}}</span>
            </div>
          </div>
          <div>
            <el-button style="padding: 3px 0; margin-left: 330px" type="primary" round v-show="!showResult" @click="uploadServer">开始评估测验</el-button>
            <el-button style="padding: 3px 0; float: right" type="primary" round v-show="showResult" @click="showMore">查看更多结果</el-button>
          </div>
        </el-card>
      </div>
      <div id="info_patient">
        <!-- 卡片放置表格 -->
        <el-card style="border-radius: 8px">
          <div slot="header" class="clearfix">
            <span>检测结果</span>
            <el-button
              style="margin-left: 35px"
              v-show="showResult"
              type="primary"
              class="download_bt"
              @click="reload"
            >
              重新上传
              <input
                ref="upload2"
                style="display: none"
                name="file"
                type="file"
              />
            </el-button>
          </div>
          <el-tabs v-model="activeName">
            <el-tab-pane label="扰动图片所检测到的目标" name="first">
              <!-- 表格存放特征值 -->
              <el-table
                :data="feature_list"
                height="620"
                style="width: 750px; text-align: center"
                v-loading="loading"
                element-loading-text="数据正在处理中，请耐心等待"
                element-loading-spinner="el-icon-loading"
                lazy
              >
                <el-table-column label="目标类别" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[0] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="目标大小" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[1] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="置信度" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[2] }}</span>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>
            <el-tab-pane label="评估各项指标" name="second">
              <!-- 表格存放特征值 -->
              <el-table
                :data="metrics_list"
                height="620"
                style="width: 750px; text-align: center"
                v-loading="loading"
                element-loading-text="数据正在处理中，请耐心等待"
                element-loading-spinner="el-icon-loading"
                lazy
              >
                <el-table-column label="扰动配置" width="150px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[0] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="平均时间/NMS时间(ms)" width="220px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[1] }} / {{ scope.row[2] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="进入NMS个数/NMS之后个数" width="280px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[3] }} / {{ scope.row[4] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="召回率(%)" width="150px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[5] }}</span>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </div>
    </div>

    <el-dialog
      title="展示更多图片结果"
      :visible.sync="showMoreDialog"
      width="80%"
      :show-close="false"
      :close-on-press-escape="false"
      :append-to-body="true"
      :close-on-click-modal="false"
      :center="true"
    >

      <el-table
        :data="img_url_list"
        height="650"
        style="text-align: center"
        v-loading="loading"
        element-loading-text="数据正在处理中，请耐心等待"
        element-loading-spinner="el-icon-loading"
        lazy
      >
        <el-table-column label="原始图像">
          <template slot-scope="scope">
            <span>
              <el-image
                :src="scope.row[0]"
                class="image_2"
                style="border-radius: 3px 3px 0 0"
              >
                <div slot="error">
                  <div slot="placeholder" class="error">加载图片失败</div>
                </div>
              </el-image>  
            </span>
          </template>
        </el-table-column>
        <el-table-column label="干净结果">
          <template slot-scope="scope">
            <span>
              <el-image
                :src="scope.row[1]"
                class="image_2"
                style="border-radius: 3px 3px 0 0"
              >
                <div slot="error">
                  <div slot="placeholder" class="error">加载图片失败</div>
                </div>
              </el-image>  
            </span>
          </template>
        </el-table-column>
        <el-table-column label="扰动结果">
          <template slot-scope="scope">
            <span>
              <el-image
                :src="scope.row[2]"
                class="image_2"
                style="border-radius: 3px 3px 0 0"
              >
                <div slot="error">
                  <div slot="placeholder" class="error">加载图片失败</div>
                </div>
              </el-image>  
            </span>
          </template>
        </el-table-column>
      </el-table>

      <span slot="footer">
        <el-button @click="showMoreDialog = false">取 消</el-button>
      </span>
    </el-dialog>

  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "Content",
  data() {
    return {
      server_url: "http://127.0.0.1:5000",
      activeName: "first",
      active: 0,
      url_1: "",
      url_2: "",
      srcList: [],
      srcList1: [],
      feature_list: [],
      metrics_list: [],
      img_url_list: [],
      wait_return: "等待上传",
      loading: false,
      showResult: false,
      percentage: 0,
      opacitys: {
        opacity: 0,
      },
      dialogTableVisible: false,
      showMoreDialog: false,
      fileListModel: [],
      fileListImage: [],
    };
  },
  created: function () {
    document.title = "YOLOv5目标检测WEB端";
  },
  methods: {
    next() {
      this.active++;
    },
    // 获得目标文件
    getObjectURL(file) {
      var url = null;
      if (window.createObjcectURL != undefined) {
        url = window.createOjcectURL(file);
      } else if (window.URL != undefined) {
        url = window.URL.createObjectURL(file);
      } else if (window.webkitURL != undefined) {
        url = window.webkitURL.createObjectURL(file);
      }
      return url;
    },
    async uploadServer(){
      this.url_1 = "";
      this.url_2 = "";
      this.srcList = [];
      this.srcList1 = [];
      this.wait_return = "";
      this.feature_list = [];

      let model = this.fileListModel.length != 0 ? this.fileListModel[0] : null
      let dataset = this.fileListImage.length != 0 ? this.fileListImage[0] : null
      
      let cancel = false
      if(model == null){
        await this.$confirm(`未检测到模型，是否使用系统默认模型？`).catch(()=>{
          this.$message({
            type: 'info',
            message: '已取消上传服务器'
          });
          cancel = true
        })
        if(cancel){
          return
        }
        model = new File(['null'], 'model.txt', {type: 'text/plain'});
      }
      
      cancel = false
      var flag = false  
      if(dataset == null){
        await this.$confirm(`未检测到数据集，是否使用系统默认数据集？`).catch(()=>{
          this.$message({
            type: 'info',
            message: '已取消上传服务器'
          });
          cancel = true
        })
        if(cancel){
          return
        }
        dataset = new File(['null'], 'dataset.txt', {type: 'text/plain'});
        flag = true
      }

      this.percentage = 0;
      this.dialogTableVisible = true;
      this.loading = true;

      let param = new FormData(); //创建form对象
      //通过append向form对象添加数据
      param.append("file", dataset); 
      param.append("file", model); 

      var time = flag ? 800 : 18 * dataset.size / 7000
      var timer = setInterval(() => {
        this.myFunc();
      }, time);
      let config = {
        headers: { "Content-Type": "multipart/form-data" },
      }; //添加请求头

      axios
        .post(this.server_url + "/upload", param, config)
        .then((response) => {
          this.percentage = 100;
          clearInterval(timer);
          this.url_1 = response.data.clean_urls[0];
          this.srcList.push(this.url_1);
          this.url_2 = response.data.draw_urls[0];
          this.srcList1.push(this.url_2);

          this.metrics_list = response.data.metrics
          this.feature_list = response.data.image_info

          for (var i = 0; i < response.data.image_urls.length; i++) {
            this.img_url_list.push([response.data.image_urls[i], response.data.clean_urls[i], response.data.draw_urls[i]])
          }

          this.dialogTableVisible = false;
          this.percentage = 0;
          this.notice1();
          this.loading = false;
          this.showResult = true;
        }).catch((response) => {
          this.$message.warning(`${response}`);
        });
    },
    showMore() {
      this.showMoreDialog = true
    },
    httpRequestModel(param){
      this.fileListModel.push(param.file)
    },
    httpRequestImage(param){
      this.fileListImage.push(param.file)
    },
    myFunc() {
      if (this.percentage + 5 < 99) {
        this.percentage = this.percentage + 5;
      } else {
        this.percentage = 99;
      }
    },
    init() {},
    notice1() {
      this.$notify({
        title: "预测成功",
        message: "点击图片可以查看大图或点击查看更多结果",
        // duration: 0,
        type: "success",
      });
    },
    reload() {
      window.location.reload();
    },
    handleExceed() {
      this.$message.warning(`限制选择 1 个文件`);
    },
    handleExceedImage() {
      this.$message.warning(`限制选择 1 个文件,如有多张请压缩成zip格式`);
    },
    beforeRemove(file) {
      return this.$confirm(`确定移除 ${ file.name }？`).then(() => {
        this.fileListModel.pop()
      });
    },
    beforeRemoveImage(file) {
      return this.$confirm(`确定移除 ${ file.name }？`).then(() => {
        this.fileListImage.pop()
      });
    }
  },
  mounted() {
    this.init();
  },
};
</script>

<style>
.el-button {
  padding: 12px 20px !important;
}

#hello p {
  font-size: 15px !important;
  /*line-height: 25px;*/
}

.n1 .el-step__description {
  padding-right: 20%;
  font-size: 14px;
  line-height: 20px;
  /* font-weight: 400; */
}
</style>

<style scoped>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

.dialog_info {
  margin: 20px auto;
}

.text {
  font-size: 14px;
}

.item {
  margin-bottom: 18px;
}

.clearfix:before,
.clearfix:after {
  display: table;
  content: "";
}

.clearfix:after {
  clear: both;
}

.box-card {
  width: 680px;
  height: 200px;
  border-radius: 8px;
  margin-top: -20px;
}

.divider {
  width: 50%;
}

#CT {
  display: flex;
  height: 100%;
  width: 100%;
  flex-wrap: wrap;
  justify-content: center;
  margin: 0 auto;
  margin-right: 0px;
  max-width: 1800px;
}

#CT_image_1 {
  width: 90%;
  height: 40%;
  margin: 0px auto;
  padding: 0px auto;
  margin-right: 180px;
  margin-bottom: 0px;
  border-radius: 4px;
}

#CT_image {
  margin-bottom: 60px;
  margin-left: 30px;
  margin-top: 5px;
}

.image_1 {
  width: 275px;
  height: 260px;
  background: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.image_2 {
  width: 500px;
  height: 460px;
  background: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.img_info_1 {
  height: 30px;
  width: 275px;
  text-align: center;
  background-color: #21b3b9;
  line-height: 30px;
}

.demo-image__preview1 {
  width: 250px;
  height: 290px;
  margin: 20px 60px;
  float: left;
}

.demo-image__preview2 {
  width: 250px;
  height: 290px;

  margin: 20px 460px;
  /* background-color: green; */
}

.error {
  margin: 100px auto;
  width: 50%;
  padding: 10px;
  text-align: center;
}

.block-sidebar {
  position: fixed;
  display: none;
  left: 50%;
  margin-left: 600px;
  top: 350px;
  width: 60px;
  z-index: 99;
}

.block-sidebar .block-sidebar-item {
  font-size: 50px;
  color: lightblue;
  text-align: center;
  line-height: 50px;
  margin-bottom: 20px;
  cursor: pointer;
  display: block;
}

div {
  display: block;
}

.block-sidebar .block-sidebar-item:hover {
  color: #187aab;
}

.download_bt {
  padding: 10px 16px !important;
}

#upfile {
  width: 104px;
  height: 45px;
  background-color: #187aab;
  color: #fff;
  text-align: center;
  line-height: 45px;
  border-radius: 3px;
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.1), 0 2px 2px 0 rgba(0, 0, 0, 0.2);
  color: #fff;
  font-family: "Source Sans Pro", Verdana, sans-serif;
  font-size: 0.875rem;
}

.file {
  width: 200px;
  height: 130px;
  position: absolute;
  left: -20px;
  top: 0;
  z-index: 1;
  -moz-opacity: 0;
  -ms-opacity: 0;
  -webkit-opacity: 0;
  opacity: 0; /*css属性&mdash;&mdash;opcity不透明度，取值0-1*/
  filter: alpha(opacity=0);
  cursor: pointer;
}

#upload {
  position: relative;
  margin: 0px 0px;
}

#Content {
  width: 85%;
  height: 800px;
  background-color: #ffffff;
  margin: 15px auto;
  display: flex;
  min-width: 1200px;
}

.divider {
  background-color: #eaeaea !important;
  height: 2px !important;
  width: 100%;
  margin-bottom: 50px;
}

.divider_1 {
  background-color: #ffffff;
  height: 2px !important;
  width: 100%;
  margin-bottom: 20px;
  margin: 20px auto;
}

.steps {
  font-family: "lucida grande", "lucida sans unicode", lucida, helvetica,
    "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
  color: #21b3b9;
  text-align: center;
  margin: 15px auto;
  font-size: 20px;
  font-weight: bold;
  text-align: center;
}

.step_1 {
  /*color: #303133 !important;*/
  margin: 20px 26px;
}

#info_patient {
  margin-top: 10px;
  margin-right: 160px;
}
</style>


