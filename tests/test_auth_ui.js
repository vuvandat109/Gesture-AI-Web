// Simulated DOM unit test; not a browser end-to-end test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
function el(id){if(!elements.has(id)){const classes=new Set(id==='authOverlay'?['hidden']:[]);elements.set(id,{value:'',textContent:'',style:{},disabled:false,content:'csrf0',lastElementChild:{textContent:''},classList:{add(...xs){xs.forEach(x=>classes.add(x))},remove(...xs){xs.forEach(x=>classes.delete(x))},toggle(x,on){if(on===undefined)on=!classes.has(x);on?classes.add(x):classes.delete(x)},contains(x){return classes.has(x)}},focus(){},addEventListener(){},replaceChildren(){},appendChild(){},append(){},click(){}})}return elements.get(id)}
let requested=[];
const ctx={console,window:{__INITIAL_USER__:null},document:{getElementById:el,querySelector:()=>el('meta'),addEventListener(){},createElement:()=>el('new')},FormData:class{},fetch:async(path)=>{requested.push(path);return {ok:true,json:async()=>path.endsWith('register')?{registered:true,next:'login',username:'demo',csrf:'csrf1'}:{username:'demo',csrf:'csrf2'}}}};
vm.createContext(ctx);vm.runInContext(fs.readFileSync('static/app.js','utf8'),ctx);
(async()=>{
vm.runInContext("authOpen('register')",ctx);el('authUsername').value='demo';el('authPassword').value='password123';
await el('authForm').onsubmit({preventDefault(){}});
assert.equal(vm.runInContext('mode',ctx),'login');assert.equal(vm.runInContext('username',ctx),null);assert.equal(el('authOverlay').classList.contains('hidden'),false);assert.equal(el('authPassword').value,'');assert.equal(el('authUsername').value,'demo');assert.match(el('authError').textContent,/Đăng ký thành công/);assert.equal(el('authSubmit').disabled,false);
el('authPassword').value='password123';await el('authForm').onsubmit({preventDefault(){}});assert.equal(vm.runInContext('username',ctx),'demo');assert.equal(el('authOverlay').classList.contains('hidden'),true);assert.deepEqual(requested,['/api/register','/api/login']);console.log('PASS U01 register switches to login; U02 login closes dialog');
})().catch(e=>{console.error(e);process.exit(1)});
