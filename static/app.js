'use strict';
const $=id=>document.getElementById(id);
let username=window.__INITIAL_USER__, selectedFile=null, mode='login', csrf=document.querySelector('meta[name="csrf-token"]').content;
function setUser(name){username=name;$('userName').textContent=name||'Khách';$('avatar').textContent=name?name.slice(0,2).toUpperCase():'GD';$('loginMenu').classList.toggle('hidden',!!name);$('registerMenu').classList.toggle('hidden',!!name);$('logoutMenu').classList.toggle('hidden',!name)}
setUser(username);
$('accountToggle').onclick=()=>{$('accountMenu').classList.toggle('hidden')};
document.addEventListener('click',e=>{if(!e.target.closest('.user-area'))$('accountMenu').classList.add('hidden')});
function authOpen(next){mode=next;$('authOverlay').classList.remove('hidden');$('accountMenu').classList.add('hidden');$('authError').textContent='';authTab(next);$('authUsername').focus()}
function authTab(tab){mode=tab;$('loginTab').classList.toggle('active',tab==='login');$('registerTab').classList.toggle('active',tab==='register');$('authTitle').textContent=tab==='login'?'Đăng nhập Gesture AI':'Tạo tài khoản Gesture AI';$('authSubmit').textContent=tab==='login'?'Đăng nhập':'Đăng ký';$('authPassword').autocomplete=tab==='login'?'current-password':'new-password'}
$('loginMenu').onclick=()=>authOpen('login');$('registerMenu').onclick=()=>authOpen('register');$('loginTab').onclick=()=>authTab('login');$('registerTab').onclick=()=>authTab('register');$('closeAuth').onclick=()=>{$('authOverlay').classList.add('hidden')};$('authOverlay').onclick=e=>{if(e.target===$('authOverlay'))$('authOverlay').classList.add('hidden')};
async function api(path,data,isFile=false){let response=await fetch(path,{method:'POST',headers:{'X-CSRF-Token':csrf,...(isFile?{}:{'Content-Type':'application/json'})},body:isFile?data:JSON.stringify(data)});let result=await response.json().catch(()=>({error:'Không đọc được phản hồi của máy chủ.'}));if(!response.ok)throw new Error(result.error||'Yêu cầu thất bại');if(result.csrf){csrf=result.csrf;document.querySelector('meta[name="csrf-token"]').content=csrf}return result}
$('authForm').onsubmit=async e=>{e.preventDefault();$('authError').textContent='';try{let result=await api('/api/'+(mode==='login'?'login':'register'),{username:$('authUsername').value,password:$('authPassword').value});setUser(result.username);$('authOverlay').classList.add('hidden');$('authPassword').value='';setMessage('Đăng nhập thành công.',false)}catch(err){$('authError').textContent=err.message}};
$('logoutMenu').onclick=async()=>{try{await api('/api/logout',{});setUser(null);$('accountMenu').classList.add('hidden');setMessage('Đã đăng xuất.',false)}catch(err){setMessage(err.message,true)}};
function useFile(file){if(!file)return;if(!/\.(csv|txt)$/i.test(file.name)){setMessage('Chỉ hỗ trợ CSV hoặc TXT.',true);return}if(file.size>2*1024*1024){setMessage('File vượt quá 2 MB.',true);return}selectedFile=file;$('fileLabel').textContent=file.name;$('fileHint').textContent=(file.size/1024).toFixed(1)+' KB';$('uploadedName').textContent='Đã chọn file dữ liệu';setMessage('',false);progress(0,'Sẵn sàng','Chọn “Chạy dự đoán” để xử lý file.')}
$('chooseBtn').onclick=()=>$('fileInput').click();$('dropzone').onclick=()=>$('fileInput').click();$('dropzone').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('fileInput').click()}};$('fileInput').onchange=e=>useFile(e.target.files[0]);for(let evt of ['dragenter','dragover'])$('dropzone').addEventListener(evt,e=>{e.preventDefault();$('dropzone').classList.add('drag')});for(let evt of ['dragleave','drop'])$('dropzone').addEventListener(evt,e=>{e.preventDefault();$('dropzone').classList.remove('drag')});$('dropzone').addEventListener('drop',e=>useFile(e.dataTransfer.files[0]));
function setMessage(s,error){$('message').textContent=s;$('message').style.color=error?'#b54747':'#348968'}
function progress(amount,status,description,type=''){ $('progressBar').style.width=amount+'%';$('progressValue').textContent=amount+'%';$('stateText').textContent=status;$('stateDetail').lastElementChild.textContent=description;$('stateBanner').classList.remove('processing','error');if(type)$('stateBanner').classList.add(type)}
const GESTURE_ICONS = new Set(['right','left','up','down','circle']);
function renderResult(result){
  $('prediction').textContent = result.prediction;
  $('confidence').textContent = result.confidence.toFixed(1) + '%';
  const key = GESTURE_ICONS.has(result.gesture_key) ? result.gesture_key : 'none';
  const icon = $('gestureImage');
  icon.src = '/static/gestures/' + key + '.svg';
  icon.alt = 'Hình minh họa cử chỉ ' + result.prediction;
  icon.classList.remove('gesture-enter');
  void icon.offsetWidth;
  icon.classList.add('gesture-enter');
  $('probabilities').replaceChildren();
  result.probabilities.forEach((p,i)=>{
    let row=document.createElement('div');row.className='prob-row'+(i===0?' primary':'');
    let name=document.createElement('span');name.className='prob-name';name.textContent=p.label;
    let track=document.createElement('div');track.className='prob-track';
    let fill=document.createElement('div');fill.className='prob-fill';
    fill.style.width=Math.max(0,Math.min(100,p.probability))+'%';track.appendChild(fill);
    let value=document.createElement('span');value.className='prob-value';value.textContent=p.probability.toFixed(1)+'%';
    row.append(name,track,value);$('probabilities').append(row);
  });
}
$('predictBtn').onclick=async()=>{if(!username){authOpen('login');return}if(!selectedFile){setMessage('Vui lòng chọn file CSV/TXT trước.',true);return}let data=new FormData();data.append('file',selectedFile);$('predictBtn').disabled=true;progress(35,'Đang xử lý','Đang trích xuất đặc trưng...','processing');setMessage('',false);try{let result=await api('/api/predict',data,true);renderResult(result);progress(100,'Đã hoàn tất',`Hoàn thành dự đoán với ${result.rows} điểm dữ liệu.`);setMessage('Đã dự đoán bằng mô hình học từ dữ liệu thật.',false)}catch(err){progress(0,'Có lỗi',err.message,'error');setMessage(err.message,true)}finally{$('predictBtn').disabled=false}};
